# base/types.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional

@dataclass
class Vector:
    x: float
    y: float

    def __init__(self, x, y=None):
        if y is None and isinstance(x, (list, tuple)):
            # Handle list/tuple input like [1, 1] or (1, 1)
            self.x = float(x[0])
            self.y = float(x[1])
        elif y is None and isinstance(x, str):
            # Handle string input
            # Remove all whitespace and unnecessary characters
            clean_str = x.replace(' ', '').strip('()[]{}')
            # Split by comma or other common separators
            coords = clean_str.split(',')
            if len(coords) == 2:
                self.x = float(coords[0])
                self.y = float(coords[1])
            else:
                raise ValueError(f"Invalid vector string format: {x}")
        elif y is None and isinstance(x, dict):
            # Handle dictionary input
            self.x = float(x.get('x', x.get(0, 0)))
            self.y = float(x.get('y', x.get(1, 0)))
        else:
            # Handle traditional x, y input
            self.x = float(x)
            self.y = float(y) if y is not None else 0.0

        # Round values as per original implementation
        self.x = round(self.x, 4)
        self.y = round(self.y, 4)

    def normalize(self) -> 'Vector':
        """normalize the vector"""

        magnitude = (self.x ** 2 + self.y ** 2) ** 0.5
        if magnitude == 0:
            return Vector(0, 0)
        return Vector(round(self.x / magnitude, 4), round(self.y / magnitude, 4))

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> 'Vector':
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> 'Vector':
        return Vector(self.x / other, self.y / other)

    def distance(self, other: 'Vector') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __eq__(self, other: 'Vector') -> bool:
        # compare two vectors with a tolerance
        return abs(self.x - other.x) < 1e-3 and abs(self.y - other.y) < 1e-3

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def dot(self, other: 'Vector') -> float:
        return round(self.x * other.x + self.y * other.y, 4)

    def cross(self, other: 'Vector') -> float:
        return round(self.x * other.y - self.y * other.x, 4)

    def length(self) -> float:
        return round(((self.x ** 2 + self.y ** 2) ** 0.5), 4)

    def __str__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"

    def __repr__(self) -> str:
        return self.__str__()


class Road:
    def __init__(self, start: Vector, end: Vector):
        self.start = start
        self.end = end
        self.direction = (end - start).normalize()
        self.length = start.distance(end)
        self.center = (start + end) / 2


class Node:
    def __init__(self, position: Vector, type: str = "normal"):
        self.position = position
        self.type = type   # "normal" or "intersection"
        self.road_name: Optional[str] = None

    def __str__(self):
        return f"Node(position={self.position}, type={self.type})"

    def __repr__(self):
        return f"Node(position={self.position}, type={self.type})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

class Edge:
    def __init__(self, node1: Node, node2: Node):
        self.node1 = node1
        self.node2 = node2
        self.weight = node1.position.distance(node2.position)

    def __str__(self):
        return f"Edge(node1={self.node1}, node2={self.node2}, distance={self.weight})"

    def __repr__(self):
        return f"Edge(node1={self.node1}, node2={self.node2}, distance={self.weight})"

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return ((self.node1.position == other.node1.position and
                self.node2.position == other.node2.position) or
                (self.node1.position == other.node2.position and
                self.node2.position == other.node1.position))

    def __hash__(self):
        if self.node1.position.x < self.node2.position.x or \
            (self.node1.position.x == self.node2.position.x and
            self.node1.position.y <= self.node2.position.y):
            pos1, pos2 = self.node1.position, self.node2.position
        else:
            pos1, pos2 = self.node2.position, self.node1.position
        return hash((pos1, pos2))
