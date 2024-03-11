### IMPLEMENTATION OF DIJKSTRA ALGORITHM FOR A POINT ROBOT ###
#=========================================================================================================================================#

## Import Necesary Packages ##
import cv2 as cv
import heapq as hq
import math
import numpy as np
import time

#----------------------------------------------------------------------------------------------------------------------------------------#

## Define Map ##
map = np.ones((500, 1200, 3), dtype='uint8')*255

# Rectangle Obstacles (Blue) with 5mm Borders (Black)
cv.rectangle(map, (97,0), (177,402), (0,0,0), thickness=3)
cv.rectangle(map, (100,0), (174,399), (255,0,0), thickness=-1)
cv.rectangle(map, (272,72), (352,502), (0,0,0), thickness=3)
cv.rectangle(map, (275,75), (349,499), (255,0,0), thickness=-1)
cv.rectangle(map, (1017,122), (1102,377), (0,0,0), thickness=3)
cv.rectangle(map, (1020,125), (1099,374), (255,0,0), thickness=-1)
cv.rectangle(map, (897,47), (1102,127), (0,0,0), thickness=3)
cv.rectangle(map, (900,50), (1099,124), (255,0,0), thickness=-1)
cv.rectangle(map, (897,372), (1102,452), (0,0,0), thickness=3)
cv.rectangle(map, (900,375), (1099,449), (255,0,0), thickness=-1)

# Boundary walls 5mm (Black)
cv.rectangle(map, (0,0), (1200,500), (0,0,0), thickness=7)

# Hexagonal Polygon (Blue) with 5mm Border (Black)
pts = np.array([[650,100],[780,175],[780,325],[650,400],[520,325],[520,175]], np.int32)
cv.fillPoly(map,[pts],(255,0,0))
pts1 = np.array([[650,97],[783,172],[783,328],[650,403],[517,328],[517,172]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(map,[pts],True,(0,0,0),thickness=3)

#----------------------------------------------------------------------------------------------------------------------------------------#

## Define a 'Node' class to store all the node informations ##
class Node():
    def __init__(self, coc=None, parent=None, free=False, closed=False):
        # Cost of Coming from 'source' node 
        self.coc = coc
        # Index of Parent node
        self.parent = parent
        # Boolean variable that denotes (True) if the node is in 'Free Space'
        self.free = free
        # Boolean variable that denotes (True) if the node is 'closed'
        self.closed = closed

#----------------------------------------------------------------------------------------------------------------------------------------#

## Initiate an array of all possible nodes from the 'map' ##
nodes = np.zeros((map.shape[0], map.shape[1]), dtype=Node)
for row in range(nodes.shape[0]):
    for col in range(nodes.shape[1]):
        nodes[row][col] = Node()
        # If the node index is in the 'Free Space' of 'map', assign (True)
        if map[row][col][2] == 255:
            nodes[row][col].free = True
            continue

#----------------------------------------------------------------------------------------------------------------------------------------#

## Define a 'Back-Tracking' function to derive path from 'source' to 'goal' node ##
def backTrack(x,y):
    print("Backtracking!!")
    track = []
    while True:
        track.append((y,x))
        if nodes[y][x].parent == None:
            track.reverse()
            break
        y,x = nodes[y][x].parent
    print("Path created!")
    return track

#----------------------------------------------------------------------------------------------------------------------------------------#

## Get 'Source' and 'Goal' node and check if it's reachable ##
while True:
    x1 = int(input("X - Coordinate of Source Node: "))
    y1 = int(input("Y - Coordinate of Source Node: "))
    x2 = int(input("X - Coordinate of Goal Node: "))
    y2 = int(input("Y - Coordinate of Goal Node: "))

    # Check if the given coordinates are in the 'Free Space'
    if nodes[500-y1][x1].free and nodes[500-y2][x2].free:
        print("Executing path planning for the given coordinates........!!!")
        y1 = 500-y1
        y2 = 500-y2
        break
    else:
        print("The given coordinates are not reachable. Try again with different coordinates")

#----------------------------------------------------------------------------------------------------------------------------------------#

## Create a copy of map to store the search state for every 500 iterations ##
img = map.copy()
# # Mark 'source' and 'goal' nodes on the 'img'
cv.circle(img,(x1,y1),4,(0,255,255),-1) # Source --> 'Yellow'
cv.circle(img,(x2,y2),4,(255,0,255),-1) # Goal --> 'Purple'
# Write out to 'dijkstra_output.avi' video file
out = cv.VideoWriter('dijkstra_output.avi', cv.VideoWriter_fourcc(*'XVID'), 60, (1200,500))
out.write(img)

#----------------------------------------------------------------------------------------------------------------------------------------#

## Define a function to search all the nodes from 'source' to 'goal' node using Dijkstra's Search ## 

# Initiate a Priority Queue / Heap Queue with updatable priorities to store all the currently 'open nodes' for each iteration 
open_nodes = []

iterations = 0
start = time.time()
while True:

    iterations += 1
    # Change the color of all pixels explored to 'green', except 'source' and 'goal' colors
    if not 0 in (img[y1][x1][0],img[y1][x1][1]):
        img[y1][x1] = (0,255,0)
    # Write search state 'img' for every 500 iterations
    if iterations/500 == iterations//500:
        out.write(img)

    # 'nodes[y1][x1]' --> current 'open' node
    if nodes[y1][x1].parent == None:
        # Cost to come for the source node is '0' itself
        nodes[y1][x1].coc = 0
        
    # Verify if the current 'open' node is 'goal' node
    if (x1,y1) == (x2,y2):
        print("Path Planning Successfull!!!")
        # Call 'Back-Tracking' function
        path = backTrack(x2,y2)
        break

    # If the current 'node' is not the 'goal' node, 'close' the node and explore neighbouring nodes
    else:
        # Close the node and explore corresponding neighbours
        nodes[y1][x1].closed = True

        # Perform All Possible Action Sets from: {((1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1))}
        # Get neighbouring nodes to the current 'open' node and add it to the Heap Queue 'open_nodes'
        
        # Initiate a list to iterate over 'actions' sets {UP, RIGHT, DOWN, LEFT}
        actions = []
        # Action 1: UP (Cost=1)
        # new node coordinates (x = x1, y = y1-1)
        actions.append((y1-1, x1, 1))
        # Action 2: RIGHT (Cost=1)
        # new node coordinates (x = x1+1, y = y1)
        actions.append((y1, x1+1, 1))
        # Action 3: DOWN (Cost=1)
        # new node coordinates (x = x1, y = y1+1)
        actions.append((y1+1, x1, 1))
        # Action 4: LEFT (Cost=1)
        # new node coordinates (x = x1-1, y = y1)
        actions.append((y1, x1-1, 1))
        # Action 5: UP-RIGHT (Cost=1.4)
        # new node coordinates (x = x1+1, y = y1-1)
        sqrt2 = 1.4 #math.sqrt(2) 
        actions.append((y1-1, x1+1, sqrt2))
        # Action 6: DOWN-RIGHT (Cost=1.4)
        # new node coordinates (x = x1+1, y = y1+1)
        actions.append((y1+1, x1+1, sqrt2))
        # Action 7: UP-LEFT (Cost=1.4)
        # new node coordinates (x = x1-1, y = y1-1)
        actions.append((y1-1, x1-1, sqrt2))
        # Action 8: DOWN-LEFT (Cost=1.4)
        # new node coordinates (x = x1-1, y = y1+1)
        actions.append((y1+1, x1-1, sqrt2))

        # Cost to come of the current open node (y1,x1)
        dist = nodes[y1][x1].coc

        # Iterate over 'actions' list with 'cost'
        for action in actions:
            y = action[0]
            x = action[1]
            cost = action[2]
            # If the neighbour node is already 'closed', iterate over next action
            if nodes[y][x].closed:
                continue
            # Check if new node is in 'Free Space'
            if nodes[y][x].free:
                # If the new node is visited for the first time, update '.coc' and '.parent'
                if nodes[y][x].coc == None:
                    nodes[y][x].coc = dist + cost
                    nodes[y][x].parent = (y1,x1)
                    # Add new node to 'open_nodes'
                    hq.heappush(open_nodes, (nodes[y][x].coc, (y, x)))
                # If the new node was already visited, update '.coc' and '.parent' only if the new_node.coc is less than the existing value
                elif (dist + cost) < nodes[y][x].coc:
                    nodes[y][x].coc = dist + cost
                    nodes[y][x].parent = (y1,x1)
                    # Update 'priority' of new node in 'open_nodes'
                    hq.heappush(open_nodes, (nodes[y][x].coc, (y, x)))

        while True:
            # Pop next element from 'open_nodes'
            (priority, node) = hq.heappop(open_nodes)
            y = node[0]
            x = node[1]
            # If priority is greater than node.coc, pop next node
            if priority == nodes[y][x].coc and nodes[y][x].closed == False:
                break

        # Update x1 and y1 for next iteration
        y1 = y
        x1 = x
# Write last frame to video file
out.write(img)

#----------------------------------------------------------------------------------------------------------------------------------------#

end = time.time()
runntime = end-start
print("Overall Execution Time: ",runntime)

#----------------------------------------------------------------------------------------------------------------------------------------#

# Iterate over 'optimalPath' and change each pixel in path to 'Red'
count = 0
for i in path:
    cv.circle(img,(i[1],i[0]),1,(0,0,255),-1)
    # Write to video file for every 500 iterations
    if count/500 == count//500:
       out.write(img)

# Last frame in path travelling
out.write(img)

# Display 'Optimal Path' for 30 seconds
cv.imshow("Optimal Path", img)
cv.waitKey(30*1000)

out.release()

#=========================================================================================================================================#
