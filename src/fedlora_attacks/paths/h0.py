"""Zero-dimensional merge events for a finite graph of evaluated model paths.

An edge enters only when its whole evaluated path is below the threshold. This is a
sampled-path view, not a certificate about the unsampled parameter space.
"""

from __future__ import annotations


def sampled_merge_events(
    vertex_losses: dict[str, float],
    edges: list[dict],
) -> list[dict]:
    """Return births and component merges in increasing clean-loss threshold order."""
    parent = {vertex: vertex for vertex in vertex_losses}

    def root(vertex: str) -> str:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    events = [
        {"kind": "birth", "threshold": float(loss), "component": vertex}
        for vertex, loss in vertex_losses.items()
    ]
    events.sort(key=lambda event: (event["threshold"], event["component"]))
    ordered_edges = sorted(edges, key=lambda edge: (float(edge["maximum_loss"]), edge["name"]))
    for edge in ordered_edges:
        left, right = edge["u"], edge["v"]
        if left not in parent or right not in parent:
            raise ValueError("edge names a vertex absent from vertex_losses")
        threshold = max(
            float(edge["maximum_loss"]), vertex_losses[left], vertex_losses[right]
        )
        a, b = root(left), root(right)
        if a != b:
            parent[b] = a
            events.append(
                {"kind": "merge", "threshold": threshold, "components": [a, b],
                 "edge": edge["name"]}
            )
    return sorted(events, key=lambda event: (event["threshold"], event["kind"] == "merge"))
