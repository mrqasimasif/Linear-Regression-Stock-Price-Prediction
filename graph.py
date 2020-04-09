graph = {"a": ["c"],
         "b": ["c", "e"],
         "c": ["a", "b", "d", "e"],
         "d": ["c"],
         "e": ["c", "b"],
         "f": []
         }

print(graph)


def AddNode(graph, parent, child):
    graph[parent].append(child)


AddNode(graph, 'f', 'g')

print(graph)
