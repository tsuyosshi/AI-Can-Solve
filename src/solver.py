import numpy as np
import queue
import heapq
from typing import List, Tuple


class State:

    def __init__(self, numbers: List[int], W: int, H: int) -> None:

        self.numbers = numbers.copy()
        self.W = W
        self.H = H
        self.path = []

        for id in range(len(numbers)):
            if numbers[id] == -1:
                self.empty_block = id
                break

        self.pre_empty_block = None

        self.score = self.calculateScore()

    def __lt__(self, other):
        # 降順に取り出したいので逆にしている
        return self.score > other.score

    def calculateScore(self) -> int:
        score = 0
        for id in range(len(self.numbers)):
            if self.numbers[id] == id+1:
                score = score+1
        return score

    def getIdFromPos(self, x: int, y: int) -> int:
        return y*self.H + x
    
    def getPosFromId(self, id: int) -> Tuple[int, int]:
        return id%self.W, id//self.H
    
    def update(self, id: int) -> None:
        self.numbers[self.empty_block] = self.numbers[id]
        self.numbers[id] = -1
        self.pre_empty_block = self.empty_block
        self.empty_block = id
        self.score = self.calculateScore()
        self.path.append(id)

    def getNextState(self, id):
        nextState = State(self.numbers, self.W, self.H)
        nextState.numbers[self.empty_block] = self.numbers[id]
        nextState.numbers[id] = -1
        nextState.pre_empty_block = self.empty_block
        nextState.empty_block = id
        nextState.score = nextState.calculateScore()
        nextState.path = self.path.copy()
        nextState.path.append(id)
        return nextState

    def updatable(self, id: int) -> bool:

        if id == self.empty_block:
            return False
        
        x, y = self.getPosFromId(id)
        ex, ey = self.getPosFromId(self.empty_block)
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]

        for i in range(4):
            nx = ex + dx[i]
            ny = ey + dy[i]
            if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
                continue
            if nx == x and ny == y:
                return True
            
        return False
    
    def backState(self) -> None:
        self.update(self.pre_empty_block)
        self.path = self.path[:-2]

    def checkComplete(self) -> bool:
        for i in range(len(self.numbers)-1):
            if self.numbers[i] == -1 or self.numbers[i]-1 != i:
                return False
        return True 

class DFS:

    def __init__(self, numbers: List[int], W: int, H: int) -> None:
        self.initState = State(numbers, W, H)
        self.path = None
        self.visited = {}

    def run(self) -> List[int]:
        self.solve(self.initState)
        return self.path

    def solve(self, state: State) -> None:

        if state.checkComplete():
            if self.path == None or len(state.path) < len(self.path):
                self.path = state.path

        self.visited[tuple(state.numbers)] = True

        for id in range(len(state.numbers)):
            if state.updatable(id):
                state.update(id)

                if tuple(state.numbers) not in self.visited.keys():
                    print("id", id)
                    print(tuple(state.numbers), self.visited.keys())
                    self.solve(state)

                state.backState()


class BFS:

    def __init__(self, numbers: List[int], W: int, H: int) -> None:
        self.initState = State(numbers, W, H)
        self.path = None
        self.visited = {}

    def run(self) -> List[int]:
        self.solve(self.initState)
        return self.path

    def solve(self, initState: State) -> None:

        que = queue.Queue()
        que.put(initState)

        self.visited[tuple(initState.numbers)] = True
        

        while not que.empty():

            state = que.get()
            # print(state.numbers)

            if state.checkComplete():
                self.path = state.path
                break

            for id in range(len(state.numbers)):

                if state.updatable(id):

                    nextState = state.getNextState(id)

                    if tuple(nextState.numbers) not in self.visited.keys():
                        que.put(nextState)
                        self.visited[tuple(nextState.numbers)] = True

class BeamSearch:

    def __init__(self, numbers: List[int], W: int, H: int, BEAM_WIDTH:int = 10000) -> None:
        self.initState = State(numbers, W, H)
        self.path = None
        self.visited = {}
        self.BEAM_WIDTH = BEAM_WIDTH

    def run(self) -> List[int]:
        self.solve(self.initState)
        return self.path

    def solve(self, initState: State) -> None:

        heap = [initState]
        heapq.heapify(heap)

        self.visited[tuple(initState.numbers)] = True
        
        print(heap)

        depth = 0

        while len(heap) > 0:

            nextHeap = []
            heapq.heapify(nextHeap)

            for i in range(min(len(heap), self.BEAM_WIDTH)):
                state = heapq.heappop(heap)
                print("step", depth)
                print("num", state.numbers)
                print("path", state.path)
                if state.checkComplete():
                    self.path = state.path
                    return

                for id in range(len(state.numbers)):

                    if state.updatable(id):

                        nextState = state.getNextState(id)

                        if tuple(nextState.numbers) not in self.visited.keys():
                            heapq.heappush(nextHeap, nextState)
                            self.visited[tuple(nextState.numbers)] = True

            heap = []
            heapq.heapify(heap)

            for i in range(min(len(nextHeap), self.BEAM_WIDTH)):
                state = heapq.heappop(nextHeap)
                heapq.heappush(heap, state)
            
            depth = depth + 1
            

numbers = [1, 2, 3, 4,
           12, 13, 14, 5,
           11, -1, 15, 6,
           10, 9, 8, 7]
W = 4
H = 4

solver = BeamSearch(numbers, W, H)
print(solver.run())







    

    


    
