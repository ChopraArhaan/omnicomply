"""Neo4j-backed knowledge graph store for regulatory entity & relation management.

Stores documents, entities, and regulation linkages in a property graph
and supports multi-hop traversal queries useful for regulatory cross-reference
analysis.
"""

from __future__ import annotations

from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """An extracted named entity to be stored in the knowledge graph.

    Attributes:
        text: Surface text of the entity.
        label: NER label (e.g. ``ORG``, ``REGULATION``, ``SECTION``).
        properties: Arbitrary extra properties attached to the node.
    """

    text: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """A node returned from a graph query.

    Attributes:
        id: Neo4j element ID or application-level identifier.
        labels: Set of Neo4j labels on the node.
        properties: All node properties.
    """

    id: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """An edge (relationship) returned from a graph query.

    Attributes:
        source: Source node identifier.
        target: Target node identifier.
        relation: Relationship type string.
        properties: Relationship properties.
    """

    source: str
    target: str
    relation: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphResult(BaseModel):
    """Result container for graph traversal queries.

    Attributes:
        nodes: Unique nodes discovered during traversal.
        edges: Relationships connecting those nodes.
        paths: Raw path representations (list of node-id sequences).
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge graph store
# ---------------------------------------------------------------------------


class KnowledgeGraphStore:
    """Async Neo4j knowledge-graph store for regulatory compliance data.

    Manages documents, named entities, and regulation-to-regulation links
    and exposes multi-hop traversal queries that power the graph component
    of the hybrid retriever.

    Args:
        uri: Neo4j Bolt URI (e.g. ``"bolt://localhost:7687"``).
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name (``"neo4j"`` by default).
    """

    # ------------------------------------------------------------------
    # Cypher query templates
    # ------------------------------------------------------------------

    _CREATE_DOCUMENT: str = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d += $metadata, d.updated_at = datetime()
    """

    _CREATE_ENTITY: str = """
        MERGE (e:Entity {text: $text, label: $label})
        SET e += $properties
        WITH e
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (d)-[:CONTAINS]->(e)
    """

    _LINK_ENTITIES: str = """
        MATCH (a:Entity {text: $source_text})
        MATCH (b:Entity {text: $target_text})
        MERGE (a)-[r:RELATED_TO]->(b)
        SET r.relation = $relation
    """

    _ADD_REGULATION_LINK: str = """
        MERGE (s:Regulation {reg_id: $source_reg})
        MERGE (t:Regulation {reg_id: $target_reg})
        MERGE (s)-[r:LINKED_TO {relation: $relation}]->(t)
        SET r.updated_at = datetime()
    """

    _QUERY_RELATED: str = """
        MATCH path = (start:Entity {text: $entity})-[*1..$hops]-(related)
        WHERE related:Entity OR related:Document OR related:Regulation
        RETURN path
        LIMIT 100
    """

    _GET_REGULATION_NETWORK: str = """
        MATCH path = (r:Regulation {reg_id: $reg_id})-[*1..3]-(connected)
        RETURN path
        LIMIT 200
    """

    _ENSURE_CONSTRAINTS: str = """
        CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
    """

    _ENSURE_ENTITY_INDEX: str = """
        CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.text, e.label)
    """

    _ENSURE_REGULATION_CONSTRAINT: str = """
        CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.reg_id IS UNIQUE
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: AsyncDriver | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Neo4j async driver and create schema constraints."""
        logger.info("neo4j_connect", uri=self._uri)
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await self._ensure_schema()
        logger.info("neo4j_connected")

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            logger.info("neo4j_closed")

    async def _ensure_schema(self) -> None:
        """Create constraints and indexes if they do not exist."""
        async with self._session() as session:
            await session.run(self._ENSURE_CONSTRAINTS)
            await session.run(self._ENSURE_ENTITY_INDEX)
            await session.run(self._ENSURE_REGULATION_CONSTRAINT)
        logger.info("neo4j_schema_ensured")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add_document(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Create or update a Document node in the graph.

        Args:
            doc_id: Unique document identifier.
            metadata: Arbitrary properties to store on the node.
        """
        logger.info("kg_add_document", doc_id=doc_id)
        async with self._session() as session:
            await session.run(
                self._CREATE_DOCUMENT,
                doc_id=doc_id,
                metadata=metadata,
            )

    async def add_entities(
        self,
        doc_id: str,
        entities: list[Entity],
    ) -> None:
        """Create entity nodes and link them to a document.

        Entities with the same ``(text, label)`` are merged (idempotent).
        Each entity is connected to the parent document via a
        ``CONTAINS`` relationship.  If multiple entities share the same
        label they are linked with a ``RELATED_TO`` edge.

        Args:
            doc_id: Parent document identifier.
            entities: Named entities to insert.
        """
        if not entities:
            return

        logger.info("kg_add_entities", doc_id=doc_id, count=len(entities))

        async with self._session() as session:
            # Upsert each entity and link to document.
            for entity in entities:
                await session.run(
                    self._CREATE_ENTITY,
                    text=entity.text,
                    label=entity.label,
                    properties=entity.properties,
                    doc_id=doc_id,
                )

            # Cross-link entities that share the same label within the
            # same document (useful for co-occurrence analysis).
            by_label: dict[str, list[str]] = {}
            for entity in entities:
                by_label.setdefault(entity.label, []).append(entity.text)

            for label, texts in by_label.items():
                if len(texts) < 2:
                    continue
                for i, src in enumerate(texts):
                    for tgt in texts[i + 1 :]:
                        await session.run(
                            self._LINK_ENTITIES,
                            source_text=src,
                            target_text=tgt,
                            relation=f"CO_OCCURS_{label}",
                        )

        logger.info("kg_add_entities_done", doc_id=doc_id)

    async def add_regulation_link(
        self,
        source_reg: str,
        target_reg: str,
        relation: str,
    ) -> None:
        """Create a directed relationship between two regulation nodes.

        Args:
            source_reg: Regulation ID of the source node.
            target_reg: Regulation ID of the target node.
            relation: Semantic label for the link (e.g. ``"amends"``,
                ``"supersedes"``, ``"cross_references"``).
        """
        logger.info(
            "kg_add_regulation_link",
            source=source_reg,
            target=target_reg,
            relation=relation,
        )
        async with self._session() as session:
            await session.run(
                self._ADD_REGULATION_LINK,
                source_reg=source_reg,
                target_reg=target_reg,
                relation=relation,
            )

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def query_related(
        self,
        entity: str,
        hops: int = 2,
    ) -> list[GraphResult]:
        """Traverse the graph from an entity up to *hops* relationships away.

        Args:
            entity: The starting entity text to search from.
            hops: Maximum relationship depth. Clamped to ``[1, 5]``.

        Returns:
            A list of :class:`GraphResult` objects representing the
            discovered subgraph.
        """
        hops = max(1, min(hops, 5))  # safety clamp
        logger.debug("kg_query_related", entity=entity, hops=hops)

        # Neo4j does not support parameterised relationship length, so
        # we interpolate it safely (integer-only).
        cypher = self._QUERY_RELATED.replace("$hops", str(hops))

        async with self._session() as session:
            result = await session.run(cypher, entity=entity)
            records = await result.data()

        return self._parse_paths(records)

    async def get_regulation_network(self, reg_id: str) -> dict[str, Any]:
        """Return the subgraph surrounding a regulation node.

        Args:
            reg_id: Regulation identifier.

        Returns:
            A dictionary with ``nodes``, ``edges``, and ``paths`` keys
            serialised from the traversal result.
        """
        logger.debug("kg_get_regulation_network", reg_id=reg_id)

        async with self._session() as session:
            result = await session.run(self._GET_REGULATION_NETWORK, reg_id=reg_id)
            records = await result.data()

        graph_results = self._parse_paths(records)

        # Merge all GraphResults into a single summary dict.
        all_nodes: dict[str, GraphNode] = {}
        all_edges: list[GraphEdge] = []
        all_paths: list[list[str]] = []

        for gr in graph_results:
            for node in gr.nodes:
                all_nodes[node.id] = node
            all_edges.extend(gr.edges)
            all_paths.extend(gr.paths)

        return {
            "reg_id": reg_id,
            "nodes": [n.model_dump() for n in all_nodes.values()],
            "edges": [e.model_dump() for e in all_edges],
            "paths": all_paths,
            "node_count": len(all_nodes),
            "edge_count": len(all_edges),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session(self) -> AsyncSession:
        """Return a new async session bound to the configured database."""
        if self._driver is None:
            raise RuntimeError(
                "KnowledgeGraphStore is not connected. Call connect() first."
            )
        return self._driver.session(database=self._database)

    @staticmethod
    def _parse_paths(records: list[dict[str, Any]]) -> list[GraphResult]:
        """Convert raw Neo4j path records into :class:`GraphResult` objects."""
        results: list[GraphResult] = []

        for record in records:
            path_data = record.get("path")
            if path_data is None:
                continue

            nodes: list[GraphNode] = []
            edges: list[GraphEdge] = []
            path_ids: list[str] = []

            # neo4j driver returns path as a Path object or dict depending
            # on serialisation.  Handle both.
            if hasattr(path_data, "nodes"):
                # Native Path object
                for node in path_data.nodes:
                    node_id = str(node.element_id)
                    nodes.append(
                        GraphNode(
                            id=node_id,
                            labels=list(node.labels),
                            properties=dict(node),
                        )
                    )
                    path_ids.append(node_id)

                for rel in path_data.relationships:
                    edges.append(
                        GraphEdge(
                            source=str(rel.start_node.element_id),
                            target=str(rel.end_node.element_id),
                            relation=rel.type,
                            properties=dict(rel),
                        )
                    )
            elif isinstance(path_data, list):
                # Serialised list of alternating nodes / rels
                for item in path_data:
                    if isinstance(item, dict):
                        node_id = str(item.get("element_id", item.get("id", "")))
                        nodes.append(
                            GraphNode(
                                id=node_id,
                                labels=item.get("labels", []),
                                properties={
                                    k: v
                                    for k, v in item.items()
                                    if k not in ("element_id", "id", "labels")
                                },
                            )
                        )
                        path_ids.append(node_id)

            results.append(
                GraphResult(nodes=nodes, edges=edges, paths=[path_ids] if path_ids else [])
            )

        return results
