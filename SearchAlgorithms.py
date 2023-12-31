from collections import deque

from VacuumState import VacuumState
from RomaniaState import *
from HopperState import *

from queue import PriorityQueue
def breadth_first_search(startState, use_closed_list=True) :
    search_queue = deque()
    closed_list = {}

    search_queue.append(startState)
    if use_closed_list :
        closed_list[startState] = True
    while len(search_queue) > 0 :
        next_state = search_queue.popleft()
        if next_state.is_goal() :
            print("Goal found")
            next_state.print_solution()
            return
        else :
            successors = next_state.successors()
            if use_closed_list :
                successors = [item for item in successors
                                    if item not in closed_list]
                for s in successors :
                    closed_list[s] = True
            search_queue.extend(successors)



def depth_first_search(startState, use_closed_list=True, limit=20) :
    search_queue = deque()
    closed_list = {}
    current = 0

    search_queue.append(startState)
    while len(search_queue) > 0:
        next_state = search_queue.pop()
        if next_state.is_goal():
            print("Goal found")
            next_state.print_solution()
            return
        else:
            successors = next_state.successors()
            current = current + 1
            if use_closed_list:
                successors = [item for item in successors
                              if item not in closed_list]
                for s in successors:
                    closed_list[s] = True
            search_queue.extend(successors)

## you write this.
def iterative_deepening_search(startState):
    limit = 1
    ## you write the rest


def a_star(startState, heuristic_fn, use_closed_list=True) :
    search_queue = PriorityQueue()
    closed_list = {}
    ## you do the rest.






## simulate uniform cost
def h1(s) :
    return 0

## implement this for the Mars rover.
def SLD(s1, s2) :
    pass

## SLD to Bucharest
def RomaniaSLD(s) :
    slds = {'Arad':366, 'Bucharest':0,'Craiova':166,'Dobreta':242,
    'Eforie':161, 'Fagaras':176, 'Giurgiu':77, 'Hirsova':151,
    'Iasi':226, 'Lugoj':244, 'Mehadia':241, 'Neamt':234, 'Oradea':380,
    'Pitesti':100, 'Rimnicu Vilcea':193, 'Sibiu':253, 'Timisoara':329,
    'Urziceni':80, 'Vaslui':199, 'Zerind':374}
    return slds[s.location]



if __name__ == "__main__" :
    #start = VacuumState('left',False,False)
    g = make_romania_graph()
    start = RomaniaState('Arad',g)
    #start = HopperState(0,0,0)
    breadth_first_search(start, True)
    a_star(start, h1, True)