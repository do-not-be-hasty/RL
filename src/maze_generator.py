import random


def generate(width=5, height=5, thickness=2, verbose=0):
    dirs = [(0,1), (1,0), (0,-1), (-1,0)]

    def move(v, d):
        return (v[0] + dirs[d][0], v[1] + dirs[d][1])

    class FindAndUnion:
        def __init__(self, obj):
            self.pointer = dict()
            
            for x in obj:
                self.pointer[x] = x
        
        def find(self, x):
            if self.pointer[x] == x:
                return x
            res = self.find(self.pointer[x])
            self.pointer[x] = res
            return res
        
        def union(self, x, y):
            x = self.find(x)
            y = self.find(y)
            self.pointer[x] = y

    def generate(w, h, thickness, verbose):
        result = [[0] * (2*w+1) for _ in range(2*h+1)]
        
        vertices = []
        edges = []
        
        for x in range(w):
            for y in range(h):
                vertices.append((x,y))
                
                if y != h-1:
                    edges.append(((x,y), move((x,y),0)))
                if x != w-1:
                    edges.append(((x,y), move((x,y),1)))
                if y != 0:
                    edges.append(((x,y), move((x,y),2)))
                if x != 0:
                    edges.append(((x,y), move((x,y),3)))
        
        random.seed()
        random.shuffle(edges)
        fau = FindAndUnion(vertices)
        
        for e in edges:
            v1 = fau.find(e[0])
            v2 = fau.find(e[1])
            
            if (v1 == v2):
                continue
            
            fau.union(v1, v2)

            result[2*e[0][1]+1][2*e[0][0]+1] = 1
            result[2*e[1][1]+1][2*e[1][0]+1] = 1
            result[e[0][1]+e[1][1]+1][e[0][0]+e[1][0]+1] = 1

        while True:
            # Player position
            x = random.randint(0, width-1) * 2 + 1
            y = random.randint(0, height-1) * 2 + 1

            if result[x][y] == 1:
                result[x][y] = 2
                break

        while True:
            # Target position
            x = random.randint(0, width-1) * 2 + 1
            y = random.randint(0, height-1) * 2 + 1

            if result[x][y] == 1:
                result[x][y] = 3
                break

        maze_list = []

        for row in result:
            maze_list.append('')
            for x in row:
                if x == 0:
                    maze_list[-1] += 'X' * thickness
                elif x == 1:
                    maze_list[-1] += ' ' * thickness
                elif x == 2:
                    maze_list[-1] += 'P' + ' ' * (thickness - 1)
                else:
                    maze_list[-1] += '^' + ' ' * (thickness - 1)

        if verbose > 0:
            for row in maze_list:
                print(row)

        return maze_list

    return generate(width, height, thickness, verbose)