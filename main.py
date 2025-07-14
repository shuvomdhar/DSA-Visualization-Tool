import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import threading
import random
from collections import deque
import heapq

class DataStructureVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Structures & Algorithms Visualizer")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.array = []
        self.speed = 0.5
        self.is_sorting = False
        self.stack = []
        self.queue = deque()
        self.binary_tree = None
        self.linked_list = None
        self.graph = {}
        self.graph_nodes = []
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Data Structure:").pack(side=tk.LEFT)
        self.ds_var = tk.StringVar(value="Array")
        ds_combo = ttk.Combobox(control_frame, textvariable=self.ds_var, 
                               values=["Array", "Stack", "Queue", "Binary Tree", "Linked List", "Graph"])
        ds_combo.pack(side=tk.LEFT, padx=(5, 20))
        ds_combo.bind('<<ComboboxSelected>>', self.on_ds_change)
        
        ttk.Label(control_frame, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value="Bubble Sort")
        self.algo_combo = ttk.Combobox(control_frame, textvariable=self.algo_var)
        self.algo_combo.pack(side=tk.LEFT, padx=(5, 20))
        self.update_algorithm_options()
        
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=0.5)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=2.0, 
                               variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(side=tk.LEFT, padx=(5, 20))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="Generate Data", 
                  command=self.generate_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Element", 
                  command=self.add_element).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Start", 
                  command=self.start_algorithm).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", 
                  command=self.reset).pack(side=tk.LEFT, padx=2)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
        self.generate_data()
        
    def on_ds_change(self, event=None):
        self.update_algorithm_options()
        self.reset()
        
    def update_algorithm_options(self):
        ds = self.ds_var.get()
        if ds == "Array":
            algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort", 
                         "Merge Sort", "Quick Sort", "Heap Sort", "Linear Search", "Binary Search"]
        elif ds == "Stack":
            algorithms = ["Push", "Pop", "Peek", "Display"]
        elif ds == "Queue":
            algorithms = ["Enqueue", "Dequeue", "Front", "Display"]
        elif ds == "Binary Tree":
            algorithms = ["Insert", "Delete", "Inorder", "Preorder", "Postorder", "BFS", "DFS", "Search"]
        elif ds == "Linked List":
            algorithms = ["Insert", "Delete", "Search", "Display"]
        elif ds == "Graph":
            algorithms = ["Dijkstra", "BFS", "DFS"]
        else:
            algorithms = ["Basic Operations"]
            
        self.algo_combo['values'] = algorithms
        if algorithms:
            self.algo_var.set(algorithms[0])
    
    def generate_data(self):
        ds = self.ds_var.get()
        if ds == "Array":
            size = random.randint(10, 30)
            self.array = [random.randint(1, 100) for _ in range(size)]
            self.visualize_array()
            self.status_var.set(f"Generated array of size {size}")
        elif ds == "Graph":
            self.generate_graph()
            self.visualize_graph()
            self.status_var.set("Generated connected graph")
    
    def generate_graph(self):
        # Generate a connected undirected graph with 10 nodes
        self.graph = {}
        self.graph_nodes = list(range(10))
        
        # Initialize empty adjacency list
        for i in range(10):
            self.graph[i] = {}
        
        # Create a minimum spanning tree to ensure connectivity
        visited = {0}
        while len(visited) < 10:
            src = random.choice(list(visited))
            dst = random.choice([i for i in range(10) if i not in visited])
            weight = random.randint(1, 10)
            self.graph[src][dst] = weight
            self.graph[dst][src] = weight
            visited.add(dst)
        
        # Add additional random edges (20% probability)
        for i in range(10):
            for j in range(i + 1, 10):
                if j not in self.graph[i] and random.random() < 0.2:
                    weight = random.randint(1, 10)
                    self.graph[i][j] = weight
                    self.graph[j][i] = weight
    
    def add_element(self):
        ds = self.ds_var.get()
        value = simpledialog.askinteger("Add Element", "Enter value:")
        if value is not None:
            if ds == "Array":
                self.array.append(value)
                self.visualize_array()
            elif ds == "Stack":
                self.stack.append(value)
                self.visualize_stack()
            elif ds == "Queue":
                self.queue.append(value)
                self.visualize_queue()
            elif ds == "Binary Tree":
                if self.binary_tree is None:
                    self.binary_tree = TreeNode(value)
                else:
                    self.insert_into_tree(self.binary_tree, value)
                self.visualize_tree()
            elif ds == "Linked List":
                if self.linked_list is None:
                    self.linked_list = ListNode(value)
                else:
                    self.insert_into_linked_list(value)
                self.visualize_linked_list()
            self.status_var.set(f"Added {value} to {ds}")
    
    def start_algorithm(self):
        if self.is_sorting:
            return
            
        ds = self.ds_var.get()
        algo = self.algo_var.get()
        
        if ds == "Array":
            if algo in ["Bubble Sort", "Selection Sort", "Insertion Sort", 
                       "Merge Sort", "Quick Sort", "Heap Sort"]:
                self.start_sorting()
            elif algo == "Linear Search":
                self.linear_search()
            elif algo == "Binary Search":
                self.binary_search()
        elif ds == "Stack":
            self.stack_operations(algo)
        elif ds == "Queue":
            self.queue_operations(algo)
        elif ds == "Binary Tree":
            self.tree_operations(algo)
        elif ds == "Linked List":
            self.linked_list_operations(algo)
        elif ds == "Graph":
            self.graph_operations(algo)
    
    def start_sorting(self):
        if not self.array:
            messagebox.showwarning("Warning", "Please generate an array first!")
            return
            
        self.is_sorting = True
        algo = self.algo_var.get()
        
        thread = threading.Thread(target=self.run_sorting_algorithm, args=(algo,))
        thread.daemon = True
        thread.start()
    
    def run_sorting_algorithm(self, algo):
        arr = self.array.copy()
        
        if algo == "Bubble Sort":
            self.bubble_sort(arr)
        elif algo == "Selection Sort":
            self.selection_sort(arr)
        elif algo == "Insertion Sort":
            self.insertion_sort(arr)
        elif algo == "Merge Sort":
            self.merge_sort(arr, 0, len(arr) - 1)
        elif algo == "Quick Sort":
            self.quick_sort(arr, 0, len(arr) - 1)
        elif algo == "Heap Sort":
            self.heap_sort(arr)
            
        self.array = arr
        self.is_sorting = False
        self.status_var.set(f"{algo} completed!")
    
    def bubble_sort(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.update_visualization_with_highlight(arr, [j, j + 1])
                    time.sleep(self.speed_var.get())
    
    def selection_sort(self, arr):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
                self.update_visualization_with_highlight(arr, [i, j, min_idx])
                time.sleep(self.speed_var.get() / 2)
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            self.update_visualization_with_highlight(arr, [i, min_idx])
            time.sleep(self.speed_var.get())
    
    def insertion_sort(self, arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                self.update_visualization_with_highlight(arr, [j, j + 1])
                j -= 1
                time.sleep(self.speed_var.get())
            arr[j + 1] = key
            self.update_visualization_with_highlight(arr, [j + 1])
            time.sleep(self.speed_var.get())
    
    def merge_sort(self, arr, left, right):
        if left < right:
            mid = (left + right) // 2
            self.merge_sort(arr, left, mid)
            self.merge_sort(arr, mid + 1, right)
            self.merge(arr, left, mid, right)
    
    def merge(self, arr, left, mid, right):
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            self.update_visualization_with_highlight(arr, [k])
            k += 1
            time.sleep(self.speed_var.get())
        
        while i < len(left_arr):
            arr[k] = left_arr[i]
            self.update_visualization_with_highlight(arr, [k])
            i += 1
            k += 1
            time.sleep(self.speed_var.get())
        
        while j < len(right_arr):
            arr[k] = right_arr[j]
            self.update_visualization_with_highlight(arr, [k])
            j += 1
            k += 1
            time.sleep(self.speed_var.get())
    
    def quick_sort(self, arr, low, high):
        if low < high:
            pi = self.partition(arr, low, high)
            self.quick_sort(arr, low, pi - 1)
            self.quick_sort(arr, pi + 1, high)
    
    def partition(self, arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.update_visualization_with_highlight(arr, [i, j, high])
                time.sleep(self.speed_var.get())
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.update_visualization_with_highlight(arr, [i + 1, high])
        time.sleep(self.speed_var.get())
        return i + 1
    
    def heap_sort(self, arr):
        n = len(arr)
        
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(arr, n, i)
        
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.update_visualization_with_highlight(arr, [0, i])
            time.sleep(self.speed_var.get())
            self.heapify(arr, i, 0)
    
    def heapify(self, arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.update_visualization_with_highlight(arr, [i, largest])
            time.sleep(self.speed_var.get())
            self.heapify(arr, n, largest)
    
    def linear_search(self):
        if not self.array:
            messagebox.showwarning("Warning", "Please generate an array first!")
            return
            
        target = simpledialog.askinteger("Linear Search", "Enter value to search:")
        if target is None:
            return
            
        self.is_sorting = True
        thread = threading.Thread(target=self.run_linear_search, args=(target,))
        thread.daemon = True
        thread.start()
    
    def run_linear_search(self, target):
        for i, val in enumerate(self.array):
            self.update_visualization_with_highlight(self.array, [i], iterator=True)
            time.sleep(self.speed_var.get())
            if val == target:
                self.status_var.set(f"Found {target} at index {i}")
                self.is_sorting = False
                return
        
        self.status_var.set(f"{target} not found in array")
        self.is_sorting = False
    
    def binary_search(self):
        if not self.array:
            messagebox.showwarning("Warning", "Please generate an array first!")
            return
            
        self.array.sort()
        self.visualize_array()
        
        target = simpledialog.askinteger("Binary Search", "Enter value to search:")
        if target is None:
            return
            
        self.is_sorting = True
        thread = threading.Thread(target=self.run_binary_search, args=(target,))
        thread.daemon = True
        thread.start()
    
    def run_binary_search(self, target):
        left, right = 0, len(self.array) - 1
        
        while left <= right:
            mid = (left + right) // 2
            self.update_visualization_with_highlight(self.array, [left, mid, right], iterator=True)
            time.sleep(self.speed_var.get())
            
            if self.array[mid] == target:
                self.status_var.set(f"Found {target} at index {mid}")
                self.is_sorting = False
                return
            elif self.array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        self.status_var.set(f"{target} not found in array")
        self.is_sorting = False
    
    def stack_operations(self, operation):
        if operation == "Push":
            self.add_element()
        elif operation == "Pop":
            if self.stack:
                popped = self.stack.pop()
                self.visualize_stack()
                self.status_var.set(f"Popped {popped}")
            else:
                self.status_var.set("Stack is empty!")
        elif operation == "Peek":
            if self.stack:
                self.status_var.set(f"Top element: {self.stack[-1]}")
            else:
                self.status_var.set("Stack is empty!")
        elif operation == "Display":
            self.visualize_stack()
    
    def queue_operations(self, operation):
        if operation == "Enqueue":
            self.add_element()
        elif operation == "Dequeue":
            if self.queue:
                dequeued = self.queue.popleft()
                self.visualize_queue()
                self.status_var.set(f"Dequeued {dequeued}")
            else:
                self.status_var.set("Queue is empty!")
        elif operation == "Front":
            if self.queue:
                self.status_var.set(f"Front element: {self.queue[0]}")
            else:
                self.status_var.set("Queue is empty!")
        elif operation == "Display":
            self.visualize_queue()
    
    def tree_operations(self, operation):
        if operation == "Insert":
            self.add_element()
        elif operation == "Delete":
            if self.binary_tree:
                value = simpledialog.askinteger("Delete", "Enter value to delete:")
                if value is not None:
                    self.binary_tree = self.delete_from_tree(self.binary_tree, value)
                    self.visualize_tree()
                    self.status_var.set(f"Deleted {value} from tree")
            else:
                self.status_var.set("Tree is empty!")
        elif operation == "Search":
            if self.binary_tree:
                target = simpledialog.askinteger("Search", "Enter value to search:")
                if target is not None:
                    found = self.search_tree(self.binary_tree, target)
                    self.status_var.set(f"Value {target} {'found' if found else 'not found'}")
        elif operation in ["Inorder", "Preorder", "Postorder", "BFS", "DFS"]:
            if self.binary_tree:
                result = []
                self.is_sorting = True
                thread = threading.Thread(target=self.run_tree_traversal, args=(operation, result))
                thread.daemon = True
                thread.start()
            else:
                self.status_var.set("Tree is empty!")
    
    def run_tree_traversal(self, operation, result):
        if operation == "Inorder":
            self.inorder_traversal(self.binary_tree, result)
        elif operation == "Preorder":
            self.preorder_traversal(self.binary_tree, result)
        elif operation == "Postorder":
            self.postorder_traversal(self.binary_tree, result)
        elif operation == "BFS":
            self.bfs_tree(self.binary_tree, result)
        elif operation == "DFS":
            self.dfs_tree(self.binary_tree, result)
        self.status_var.set(f"{operation} traversal: {result}")
        self.is_sorting = False
    
    def linked_list_operations(self, operation):
        if operation == "Insert":
            self.add_element()
        elif operation == "Delete":
            if self.linked_list:
                value = simpledialog.askinteger("Delete", "Enter value to delete:")
                if value is not None:
                    self.delete_from_linked_list(value)
                    self.visualize_linked_list()
                    self.status_var.set(f"Deleted {value} from linked list")
            else:
                self.status_var.set("Linked list is empty!")
        elif operation == "Search":
            if self.linked_list:
                value = simpledialog.askinteger("Search", "Enter value to search:")
                if value is not None:
                    found = self.search_linked_list(value)
                    self.status_var.set(f"Value {value} {'found' if found else 'not found'}")
        elif operation == "Display":
            self.visualize_linked_list()
    
    def graph_operations(self, operation):
        if not self.graph:
            messagebox.showwarning("Warning", "Please generate a graph first!")
            return
            
        if operation == "Dijkstra":
            start = simpledialog.askinteger("Start Node", "Enter start node (0-9):", minvalue=0, maxvalue=9)
            goal = simpledialog.askinteger("Goal Node", "Enter goal node (0-9):", minvalue=0, maxvalue=9)
            if start is None or goal is None:
                return
            self.is_sorting = True
            thread = threading.Thread(target=self.run_graph_algorithm, args=(operation, start, goal))
            thread.daemon = True
            thread.start()
        elif operation in ["BFS", "DFS"]:
            start = simpledialog.askinteger("Start Node", "Enter start node (0-9):", minvalue=0, maxvalue=9)
            if start is None:
                return
            self.is_sorting = True
            thread = threading.Thread(target=self.run_graph_traversal, args=(operation, start))
            thread.daemon = True
            thread.start()
    
    def run_graph_algorithm(self, algo, start, goal):
        if algo == "Dijkstra":
            path = self.dijkstra(start, goal)
            if not path:
                self.status_var.set(f"No path found from {start} to {goal}")
            else:
                self.status_var.set(f"Dijkstra path: {path}")
            self.visualize_graph(path=path)
        self.is_sorting = False
    
    def run_graph_traversal(self, algo, start):
        result = []
        if algo == "BFS":
            self.bfs_graph(start, result)
        elif algo == "DFS":
            self.dfs_graph(start, result)
        self.status_var.set(f"{algo} traversal: {result}")
        self.is_sorting = False
    
    def dijkstra(self, start, goal):
        distances = {node: float('infinity') for node in self.graph}
        distances[start] = 0
        previous = {node: None for node in self.graph}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.visualize_graph(highlight_nodes=[current])
            time.sleep(self.speed_var.get())
            
            if current == goal:
                break
                
            for neighbor, weight in self.graph[current].items():
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        return path if path[0] == start else []
    
    def bfs_graph(self, start, result):
        if start not in self.graph:
            return
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            result.append(current)
            self.visualize_graph(highlight_nodes=[current])
            time.sleep(self.speed_var.get())
            
            for neighbor in sorted(self.graph[current].keys()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def dfs_graph(self, start, result):
        if start not in self.graph:
            return
        visited = set()
        stack = [start]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                result.append(current)
                self.visualize_graph(highlight_nodes=[current])
                time.sleep(self.speed_var.get())
                for neighbor in sorted(self.graph[current].keys(), reverse=True):
                    if neighbor not in visited:
                        stack.append(neighbor)
    
    def update_visualization_with_highlight(self, arr, highlight_indices, iterator=False):
        self.ax.clear()
        colors = ['red' if i in highlight_indices else 'skyblue' for i in range(len(arr))]
        if iterator:
            colors = ['orange' if i in highlight_indices else 'skyblue' for i in range(len(arr))]
        bars = self.ax.bar(range(len(arr)), arr, color=colors)
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Array Visualization')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{arr[i]}', ha='center', va='bottom')
        
        self.canvas.draw()
    
    def visualize_array(self):
        self.ax.clear()
        if self.array:
            bars = self.ax.bar(range(len(self.array)), self.array, color='skyblue')
            self.ax.set_xlabel('Index')
            self.ax.set_ylabel('Value')
            self.ax.set_title('Array Visualization')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{self.array[i]}', ha='center', va='bottom')
        
        self.canvas.draw()
    
    def visualize_stack(self):
        self.ax.clear()
        if self.stack:
            y_pos = range(len(self.stack))
            bars = self.ax.barh(y_pos, self.stack, color='lightgreen')
            self.ax.set_xlabel('Value')
            self.ax.set_ylabel('Stack Level')
            self.ax.set_title('Stack Visualization')
            self.ax.set_yticks(y_pos)
            self.ax.set_yticklabels([f'Level {i}' for i in range(len(self.stack))])
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                self.ax.text(width/2, bar.get_y() + bar.get_height()/2,
                            f'{self.stack[i]}', ha='center', va='center')
        
        self.canvas.draw()
    
    def visualize_queue(self):
        self.ax.clear()
        if self.queue:
            queue_list = list(self.queue)
            bars = self.ax.bar(range(len(queue_list)), queue_list, color='lightcoral')
            self.ax.set_xlabel('Position')
            self.ax.set_ylabel('Value')
            self.ax.set_title('Queue Visualization')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{queue_list[i]}', ha='center', va='bottom')
        
        self.canvas.draw()
    
    def visualize_tree(self, highlight_nodes=None):
        self.ax.clear()
        if self.binary_tree:
            nodes = self.get_tree_nodes(self.binary_tree)
            if nodes:
                levels = {}
                self.assign_levels(self.binary_tree, 0, levels)
                
                for node, level in levels.items():
                    x = node.val
                    y = -level
                    color = 'orange' if highlight_nodes and node in highlight_nodes else 'lightblue'
                    self.ax.scatter(x, y, s=500, c=color, zorder=2)
                    self.ax.text(x, y, str(node.val), ha='center', va='center', fontsize=10, zorder=3)
                
                self.draw_tree_edges(self.binary_tree, levels)
        
        self.ax.set_title('Binary Tree Visualization')
        self.canvas.draw()
    
    def visualize_linked_list(self):
        self.ax.clear()
        if self.linked_list:
            nodes = []
            current = self.linked_list
            while current:
                nodes.append(current.val)
                current = current.next
            
            if nodes:
                x_pos = range(len(nodes))
                bars = self.ax.bar(x_pos, nodes, color='plum')
                self.ax.set_xlabel('Position')
                self.ax.set_ylabel('Value')
                self.ax.set_title('Linked List Visualization')
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    self.ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{nodes[i]}', ha='center', va='bottom')
                
                for i in range(len(nodes) - 1):
                    self.ax.arrow(i + 0.5, nodes[i], 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        self.canvas.draw()
    
    def visualize_graph(self, path=None, highlight_nodes=None):
        self.ax.clear()
        if self.graph and self.graph_nodes:
            # Circular layout for nodes
            angles = np.linspace(0, 2 * np.pi, len(self.graph_nodes), endpoint=False)
            positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
            
            # Draw edges
            for node in self.graph:
                for neighbor, weight in self.graph[node].items():
                    if node < neighbor:  # Avoid drawing same edge twice
                        x_values = [positions[node][0], positions[neighbor][0]]
                        y_values = [positions[node][1], positions[neighbor][1]]
                        self.ax.plot(x_values, y_values, 'k-', alpha=0.3, zorder=1)
                        mid_x = (x_values[0] + x_values[1]) / 2
                        mid_y = (y_values[0] + y_values[1]) / 2
                        self.ax.text(mid_x, mid_y, str(weight), fontsize=8, color='blue', zorder=3)
            
            # Draw path if provided
            if path:
                for i in range(len(path) - 1):
                    x_values = [positions[path[i]][0], positions[path[i + 1]][0]]
                    y_values = [positions[path[i]][1], positions[path[i + 1]][1]]
                    self.ax.plot(x_values, y_values, 'r-', linewidth=2, alpha=0.6, zorder=2)
            
            # Draw nodes
            for node in self.graph_nodes:
                color = 'orange' if highlight_nodes and node in highlight_nodes else 'lightblue'
                self.ax.scatter(positions[node][0], positions[node][1], s=500, c=color, zorder=4)
                self.ax.text(positions[node][0], positions[node][1], str(node),
                            ha='center', va='center', fontsize=10, zorder=5)
        
        self.ax.set_title('Graph Visualization')
        self.ax.axis('equal')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.canvas.draw()
    
    def get_tree_nodes(self, node):
        if not node:
            return []
        return [node] + self.get_tree_nodes(node.left) + self.get_tree_nodes(node.right)
    
    def assign_levels(self, node, level, levels):
        if node:
            levels[node] = level
            self.assign_levels(node.left, level + 1, levels)
            self.assign_levels(node.right, level + 1, levels)
    
    def draw_tree_edges(self, node, levels):
        if node:
            if node.left:
                self.ax.plot([node.val, node.left.val], 
                           [-levels[node], -levels[node.left]], 'k-', alpha=0.6)
            if node.right:
                self.ax.plot([node.val, node.right.val], 
                           [-levels[node], -levels[node.right]], 'k-', alpha=0.6)
            
            self.draw_tree_edges(node.left, levels)
            self.draw_tree_edges(node.right, levels)
    
    def insert_into_tree(self, node, val):
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self.insert_into_tree(node.left, val)
        else:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self.insert_into_tree(node.right, val)
    
    def delete_from_tree(self, node, val):
        if not node:
            return None
            
        if val < node.val:
            node.left = self.delete_from_tree(node.left, val)
        elif val > node.val:
            node.right = self.delete_from_tree(node.right, val)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            
            successor = self.find_min(node.right)
            node.val = successor.val
            node.right = self.delete_from_tree(node.right, successor.val)
        
        return node
    
    def find_min(self, node):
        current = node
        while current.left:
            current = current.left
        return current
    
    def search_tree(self, node, val):
        if not node or node.val == val:
            return node is not None
        
        if val < node.val:
            return self.search_tree(node.left, val)
        return self.search_tree(node.right, val)
    
    def inorder_traversal(self, node, result):
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.val)
            self.visualize_tree(highlight_nodes=[node])
            time.sleep(self.speed_var.get())
            self.inorder_traversal(node.right, result)
    
    def preorder_traversal(self, node, result):
        if node:
            result.append(node.val)
            self.visualize_tree(highlight_nodes=[node])
            time.sleep(self.speed_var.get())
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
    
    def postorder_traversal(self, node, result):
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.val)
            self.visualize_tree(highlight_nodes=[node])
            time.sleep(self.speed_var.get())
    
    def bfs_tree(self, node, result):
        if not node:
            return
        queue = deque([node])
        while queue:
            current = queue.popleft()
            result.append(current.val)
            self.visualize_tree(highlight_nodes=[current])
            time.sleep(self.speed_var.get())
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
    
    def dfs_tree(self, node, result):
        if not node:
            return
        stack = [node]
        visited = set()
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                result.append(current.val)
                self.visualize_tree(highlight_nodes=[current])
                time.sleep(self.speed_var.get())
                if current.right:
                    stack.append(current.right)
                if current.left:
                    stack.append(current.left)
    
    def insert_into_linked_list(self, val):
        if not self.linked_list:
            self.linked_list = ListNode(val)
        else:
            current = self.linked_list
            while current.next:
                current = current.next
            current.next = ListNode(val)
    
    def delete_from_linked_list(self, val):
        if not self.linked_list:
            return
            
        if self.linked_list.val == val:
            self.linked_list = self.linked_list.next
            return
            
        current = self.linked_list
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
    
    def search_linked_list(self, val):
        current = self.linked_list
        while current:
            if current.val == val:
                return True
            current = current.next
        return False
    
    def reset(self):
        self.is_sorting = False
        ds = self.ds_var.get()
        
        if ds == "Array":
            self.array = []
            self.visualize_array()
        elif ds == "Stack":
            self.stack = []
            self.visualize_stack()
        elif ds == "Queue":
            self.queue = deque()
            self.visualize_queue()
        elif ds == "Binary Tree":
            self.binary_tree = None
            self.visualize_tree()
        elif ds == "Linked List":
            self.linked_list = None
            self.visualize_linked_list()
        elif ds == "Graph":
            self.graph = {}
            self.graph_nodes = []
            self.visualize_graph()
        
        self.status_var.set("Reset complete")

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

if __name__ == "__main__":
    root = tk.Tk()
    app = DataStructureVisualizer(root)
    root.mainloop()