from Data import Dataset, Example
from Tree import ID3Tree
from Node import *
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
import networkx as nx
import json
import pandas as pd
import os

class DecisionTreeApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Decision Tree System")
        self.geometry("900x650")
        
        # Variables to store data
        self.dataset = None
        self.tree = None
        self.attribute_names = []
        self.id3_tree = None
        
        # Configure notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the three main tabs
        self.create_data_input_tab()
        self.create_solution_tab()

    ############## input data part ##############
    def create_data_input_tab(self):
        """Create the data input tab"""
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Data Input")
        
        # Main frame divided into two sections
        main_frame = ttk.Frame(data_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === Section 1: CSV File Upload ===
        csv_frame = ttk.LabelFrame(main_frame, text="Load CSV File", bootstyle=PRIMARY)
        csv_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # File selection frame
        file_frame = ttk.Frame(csv_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=5)
        
        browse_btn = ttk.Button(
            file_frame, 
            text="Browse", 
            command=self.browse_csv_file,
            style="info.TButton"
        )
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = ttk.Button(
            file_frame, 
            text="Load File", 
            command=self.load_csv_file,
            style="success.TButton"
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        
        # Label to show loaded file name
        self.file_label = ttk.Label(csv_frame, text="No file selected yet")
        self.file_label.pack(padx=10, pady=5)
        
        # === Section 2: Manual Data Entry ===
        manual_frame = ttk.LabelFrame(main_frame, text="Manual Data Entry", bootstyle=PRIMARY)
        manual_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame for table dimensions
        dim_frame = ttk.Frame(manual_frame)
        dim_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Label and entry for rows
        ttk.Label(dim_frame, text="Number of Rows:").pack(side=tk.LEFT, padx=5)
        self.rows_var = tk.StringVar(value="5")
        ttk.Entry(dim_frame, textvariable=self.rows_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Label and entry for columns
        ttk.Label(dim_frame, text="Number of Columns:").pack(side=tk.LEFT, padx=5)
        self.cols_var = tk.StringVar(value="4")
        ttk.Entry(dim_frame, textvariable=self.cols_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Button to create table
        create_table_btn = ttk.Button(
            dim_frame, 
            text="Create Table", 
            command=self.create_data_table,
            bootstyle=(INFO, OUTLINE)
        )
        create_table_btn.pack(side=tk.LEFT, padx=20)
        
        # Frame for the data table
        self.table_container = ttk.Frame(manual_frame)
        self.table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button to save manually entered data
        self.save_manual_btn = ttk.Button(
            manual_frame, 
            text="Save Data", 
            command=self.save_manual_data,
            bootstyle=SUCCESS,
            state="disabled"
        )
        self.save_manual_btn.pack(pady=10)
    
    def browse_csv_file(self):
        """Open file dialog to select CSV file"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if file_path:
            self.file_path_var.set(file_path)

    def load_csv_file(self):
        """Function to load a CSV file"""
        
        file_path = self.file_path_var.get()
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist.")
            return
        
        try:
            # Read the CSV file
            self.dataset = pd.read_csv(file_path)
            self.attribute_names = list(self.dataset.columns)
            
            # Update label with file name
            file_name = file_path.split("/")[-1]
            self.file_label.config(text=f"Loaded file: {file_name}")
            
            # Display the data in the table
            self.display_loaded_data()
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"Successfully loaded {len(self.dataset)} rows and {len(self.attribute_names)} columns."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def display_loaded_data(self):
        """Display loaded CSV data in the table"""
        if self.dataset is None:
            return
        
        # Clear the table container
        for widget in self.table_container.winfo_children():
            widget.destroy()
        
        # Get dimensions of the dataset
        rows, cols = self.dataset.shape
        
        # Create frame for headers
        header_frame = ttk.Frame(self.table_container)
        header_frame.pack(fill=tk.X, pady=5)
        
        # Create empty label in top-left corner
        ttk.Label(header_frame, text="", width=4).grid(row=0, column=0)
        
        # Create labels for column headers
        self.header_entries = []
        for c, col_name in enumerate(self.attribute_names):
            header_label = ttk.Label(header_frame, text=col_name, width=15, anchor="center")
            header_label.grid(row=0, column=c+1, padx=2)
            self.header_entries.append(header_label)
        
        # Create frame with scrollbars for the table
        table_frame = ttk.Frame(self.table_container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=VERTICAL)
        y_scroll.pack(side=RIGHT, fill=Y)
        
        x_scroll = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        x_scroll.pack(side=BOTTOM, fill=X)
        
        # Create canvas to place the table
        self.table_canvas = tk.Canvas(table_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.table_canvas.pack(fill=tk.BOTH, expand=True)
        
        y_scroll.config(command=self.table_canvas.yview)
        x_scroll.config(command=self.table_canvas.xview)
        
        # Create frame inside canvas
        self.cells_frame = ttk.Frame(self.table_canvas)
        self.table_canvas.create_window((0, 0), window=self.cells_frame, anchor=NW)
        
        # Create labels to display the data
        for r in range(min(rows, 100)):  # Limit to 100 rows for performance
            # Row label
            ttk.Label(self.cells_frame, text=f"{r+1}", width=4).grid(row=r, column=0)
            
            for c in range(cols):
                cell_value = str(self.dataset.iloc[r, c])
                cell_label = ttk.Label(self.cells_frame, text=cell_value, width=15, anchor="center")
                cell_label.grid(row=r, column=c+1, padx=2, pady=2)
        
        # Update scroll region
        self.cells_frame.update_idletasks()
        self.table_canvas.config(scrollregion=self.table_canvas.bbox("all"))
        
        # Enable navigation to the solution tab
        self.notebook.select(1)
    
    def create_data_table(self):
        """Create table for manual data entry"""
        try:
            # Get input dimensions
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            
            if rows <= 0 or cols <= 0:
                raise ValueError("Dimensions must be greater than zero")
            
            # Clear the table container
            for widget in self.table_container.winfo_children():
                widget.destroy()
            
            # Create frame for headers
            header_frame = ttk.Frame(self.table_container)
            header_frame.pack(fill=tk.X, pady=5)
            
            # Create empty label in top-left corner
            ttk.Label(header_frame, text="", width=4).grid(row=0, column=0)
            
            # Create entries for column headers
            self.header_entries = []
            for c in range(cols):
                header_entry = ttk.Entry(header_frame, width=15)
                header_entry.grid(row=0, column=c+1, padx=2)
                header_entry.insert(0, f"Feature_{c+1}" if c < cols-1 else "Class")
                self.header_entries.append(header_entry)
            
            # Create frame with scrollbars for the table
            table_frame = ttk.Frame(self.table_container)
            table_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars
            y_scroll = ttk.Scrollbar(table_frame, orient=VERTICAL)
            y_scroll.pack(side=RIGHT, fill=Y)
            
            x_scroll = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
            x_scroll.pack(side=BOTTOM, fill=X)
            
            # Create canvas to place the table
            self.table_canvas = tk.Canvas(table_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            self.table_canvas.pack(fill=tk.BOTH, expand=True)
            
            y_scroll.config(command=self.table_canvas.yview)
            x_scroll.config(command=self.table_canvas.xview)
            
            # Create frame inside canvas
            self.cells_frame = ttk.Frame(self.table_canvas)
            self.table_canvas.create_window((0, 0), window=self.cells_frame, anchor=NW)
            
            # Create entries for the cells
            self.cell_entries = []
            for r in range(rows):
                row_entries = []
                # Row label
                ttk.Label(self.cells_frame, text=f"{r+1}", width=4).grid(row=r, column=0)
                
                for c in range(cols):
                    cell_entry = ttk.Entry(self.cells_frame, width=15)
                    cell_entry.grid(row=r, column=c+1, padx=2, pady=2)
                    row_entries.append(cell_entry)
                self.cell_entries.append(row_entries)
            
            # Update scroll region
            self.cells_frame.update_idletasks()
            self.table_canvas.config(scrollregion=self.table_canvas.bbox("all"))
            
            # Enable save button
            self.save_manual_btn.config(state="normal")
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def save_manual_data(self):
        """Save manually entered data"""
        try:
            # Get headers
            headers = [entry.get().strip() for entry in self.header_entries]
            
            # Verify headers are not empty
            if any(not header for header in headers):
                raise ValueError("Feature names cannot be empty")
            
            # Get cell data
            data = []
            for row in self.cell_entries:
                row_data = [entry.get() for entry in row]
                data.append(row_data)
            
            # Create DataFrame
            self.dataset = pd.DataFrame(data, columns=headers)
            self.attribute_names = headers
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"Successfully saved {len(self.dataset)} rows and {len(self.attribute_names)} columns."
            )
            
            # Activate solution tab
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")


    ############## generate solution part ##############
    def create_solution_tab(self):
        """Create the solution tab (tree building)"""
        solution_tab = ttk.Frame(self.notebook)
        self.notebook.add(solution_tab, text="Build Tree")
        
        # Create main content frame
        main_frame = ttk.Frame(solution_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Build tree button
        self.build_tree_btn = ttk.Button(
            controls_frame,
            text="Build Decision Tree",
            command=self.build_decision_tree,
            bootstyle=SUCCESS
        )
        self.build_tree_btn.pack(side=tk.LEFT, padx=5)
        
        plot_tree_btn = ttk.Button(
            controls_frame, 
            text="Plot Tree Visualization", 
            command=self.display_tree_graph,
            bootstyle=PRIMARY
        )
        plot_tree_btn.pack(side=tk.LEFT, padx=5)
        
        # Save tree button (initially disabled)
        self.save_tree_btn = ttk.Button(
            controls_frame,
            text="Save Tree as JSON",
            command=self.save_tree_json,
            bootstyle=(INFO, OUTLINE),
            state="disabled"
        )
        self.save_tree_btn.pack(side=tk.LEFT, padx=5)
        
                    # load tree button
        self.load_tree_btn = ttk.Button(
            controls_frame,
            text="Load Tree",
            command=self.load_tree_json,
            bootstyle=(WARNING, OUTLINE),
            state="disabled" #  I will enable it after I extract the Features from the tree and not from the data.
        )
        self.load_tree_btn.pack(side=tk.LEFT, padx=5)
        
        # Dataset info frame
        info_frame = ttk.LabelFrame(main_frame, text="Dataset Information")
        info_frame.pack(fill=tk.X, pady=10)
        
        # Dataset info labels
        self.dataset_info = ttk.Label(info_frame, text="No dataset loaded")
        self.dataset_info.pack(pady=5, padx=10, anchor=tk.W)
        
        self.features_info = ttk.Label(info_frame, text="Features: None")
        self.features_info.pack(pady=5, padx=10, anchor=tk.W)
        
        self.labels_info = ttk.Label(info_frame, text="Possible classes: None")
        self.labels_info.pack(pady=5, padx=10, anchor=tk.W)
        
        prediction_frame = ttk.LabelFrame(main_frame, text="Real-time Prediction")
        prediction_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Frame for input fields
        self.input_fields_frame = ttk.Frame(prediction_frame)
        self.input_fields_frame.pack(pady=10)
        
        # Button and result display
        predict_controls = ttk.Frame(prediction_frame)
        predict_controls.pack(pady=10)
        
        self.predict_btn = ttk.Button(
            predict_controls,
            text="Predict",
            command=self.run_prediction,
            bootstyle=SUCCESS,
            state="disabled"
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.prediction_result = ttk.Label(
            predict_controls,
            text="Prediction: ",
            font=("Helvetica", 12, "bold"),
            bootstyle="inverse-dark"
        )
        self.prediction_result.pack(side=tk.LEFT, padx=15)
        
        # Initialize input fields
        self.attribute_inputs = {}    
    # A new function for creating dynamic input fields
    def create_prediction_inputs(self):
        # Clear previous inputs
        for widget in self.input_fields_frame.winfo_children():
            widget.destroy()
        
        self.attribute_inputs = {}
        
        if self.dataset is not None and self.attribute_names:
            # Get features (all attributes except last one which is class)
            features = self.attribute_names[:-1]
            
            for i, feature in enumerate(features):
                # Create label
                lbl = ttk.Label(self.input_fields_frame, text=f"{feature}:")
                lbl.grid(row=i, column=0, padx=5, pady=5, sticky="e")
                
                # Create input with suggested values
                values = self.get_possible_values(feature)
                if values:
                    input_widget = ttk.Combobox(
                        self.input_fields_frame,
                        values=values,
                        width=20
                    )
                    input_widget.set("Select value")
                else:
                    input_widget = ttk.Entry(self.input_fields_frame, width=23)
                
                input_widget.grid(row=i, column=1, padx=5, pady=5)
                self.attribute_inputs[feature] = input_widget
                
            # Enable predict button if tree exists
            self.predict_btn.config(state="normal" if self.id3_tree else "disabled")        
    # Auxiliary function to obtain possible values ​​of the feature
    def get_possible_values(self, feature):
        try:
            if self.dataset is not None:
                return list(self.dataset[feature].unique())
        except KeyError:
            return []
        return []
    # Main prediction function
    def run_prediction(self):
        if self.id3_tree is None:
            messagebox.showerror("Error", "Please build the tree first!")
            return
        
        try:
            # Collect input values
            feature_values = {}
            for feature, widget in self.attribute_inputs.items():
                value = widget.get() if isinstance(widget, ttk.Combobox) else widget.get()
                if not value:
                    raise ValueError(f"Please enter value for {feature}")
                feature_values[feature] = value
            
            # Create example object
            example = Example(
                attribute_names=list(self.attribute_inputs.keys()),
                attribute_values=list(feature_values.values())
            )
            
            # Get prediction
            prediction = self.id3_tree.classify(example)
            
            # Update result display
            self.prediction_result.config(text=f"Prediction: {prediction}", bootstyle="success")
            
        except Exception as e:
            self.prediction_result.config(text=f"Error: {str(e)}", bootstyle="danger")
            messagebox.showerror("Prediction Error", str(e))    
           
    def update_dataset_info(self):
        """Update dataset information display"""
        if self.dataset is not None:
            rows, cols = self.dataset.shape
            self.dataset_info.config(text=f"Dataset: {rows} rows, {cols} columns")
            
            features = list(self.dataset.columns[:-1])
            features_text = f"Features ({len(features)}): {', '.join(features)}"
            self.features_info.config(text=features_text)
            
            # Get unique class values
            class_values = self.dataset[self.dataset.columns[-1]].unique()
            labels_text = f"Possible classes ({len(class_values)}): {', '.join(map(str, class_values))}"
            self.labels_info.config(text=labels_text)
            
            # Enable build tree button
            self.build_tree_btn.config(state="normal")
            self.create_prediction_inputs()
            self.load_tree_btn.config(state="normal" if self.dataset is not None else "disabled")
        else:
            self.dataset_info.config(text="No dataset loaded")
            self.features_info.config(text="Features: None")
            self.labels_info.config(text="Possible classes: None")
            self.build_tree_btn.config(state="disabled")
    
    def build_decision_tree(self):
        """Build the decision tree from the dataset"""
        if self.dataset is None:
            messagebox.showerror("Error", "No dataset loaded. Please load or create a dataset first.")
            return
        
        try:
            # Convert pandas DataFrame to our Dataset format
            custom_dataset = Dataset(self.dataset)
            
            # Create and build the ID3 tree
            self.id3_tree = ID3Tree(custom_dataset)
            self.id3_tree.build()

            
            # Enable save button
            self.save_tree_btn.config(state="normal")
            
            # Show success message
            messagebox.showinfo("Success", "Decision tree built successfully!")
            
            # Activate the visualization tab
            # self.notebook.select(2)
            self.create_prediction_inputs()
            self.predict_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build decision tree: {str(e)}")
            # Re-raise for debugging
            raise
    
    def save_tree_json(self):
        """Save the decision tree as a JSON file"""
        if self.id3_tree is None or self.id3_tree.tree is None:
            messagebox.showerror("Error", "No tree to save!")
            return
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            title="Save Tree as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Export tree to JSON
            self.id3_tree.export_to_json(file_path)
            messagebox.showinfo("Success", f"Tree saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save tree: {str(e)}")
    # Override the methods from part 1 that need to update the solution tab
    def load_csv_file(self):
        """Function to load a CSV file (overridden to update solution tab)"""
        
        file_path = self.file_path_var.get()
        
        if not file_path:
            return
        try:
            # Read the CSV file
            self.dataset = pd.read_csv(file_path)
            self.attribute_names = list(self.dataset.columns)
            
            # Update label with file name
            file_name = os.path.basename(file_path)
            self.file_label.config(text=f"Loaded file: {file_name}")
            
            # Display the data in the table
            self.display_loaded_data()
            
            # Update solution tab info
            self.update_dataset_info()
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"Successfully loaded {len(self.dataset)} rows and {len(self.attribute_names)} columns."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def save_manual_data(self):
        """Save manually entered data (overridden to update solution tab)"""
        try:
            # Get headers
            headers = [entry.get().strip() for entry in self.header_entries]
            
            # Verify headers are not empty
            if any(not header for header in headers):
                raise ValueError("Feature names cannot be empty")
            
            # Get cell data
            data = []
            for row in self.cell_entries:
                row_data = [entry.get() for entry in row]
                data.append(row_data)
            
            # Create DataFrame
            self.dataset = pd.DataFrame(data, columns=headers)
            self.attribute_names = headers
            
            # Update solution tab info
            self.update_dataset_info()
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"Successfully saved {len(self.dataset)} rows and {len(self.attribute_names)} columns."
            )
            
            # Activate solution tab
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def load_tree_json(self):
        """load tree from JSON file"""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree_dict = json.load(f)
            
            # convert dictionary to tree obj
            self.id3_tree = ID3Tree()
            self.id3_tree.tree = self.dict_to_tree(tree_dict)
            
            # Checking the tree's compatibility with the data
            if self.dataset is not None:
                self._validate_tree_compatibility()
            
            # Enable tree-based features
            self.save_tree_btn.config(state="normal")
            self.predict_btn.config(state="normal")
            
            
            messagebox.showinfo("Success", f"Tree loaded from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tree: {str(e)}")

    def dict_to_tree(self, tree_dict):
        """Convert JSON dictionary to tree objects"""
        if tree_dict['type'] == 'leaf':
            return Leaf(tree_dict['label'])
        
        node = Node(tree_dict['attribute'])
        for value, child_dict in tree_dict['children'].items():
            node.children[value] = self.dict_to_tree(child_dict)
        
        return node

    def _validate_tree_compatibility(self):
        """Verify that tree attributes match data"""
        if self.dataset is None:
            return

    def _get_tree_attributes(self, node):
        """Get all attributes used in the tree"""
        attributes = set()
        if isinstance(node, Node):
            attributes.add(node.attribute_tested)
            for child in node.children.values():
                attributes.update(self._get_tree_attributes(child))
        return attributes
    ############## tree visualization part ##############
    
    def hierarchy_pos(self, G, root_node=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        Calculating node locations for a hierarchical tree.
        """
        import networkx as nx
        if not nx.is_tree(G):
            raise TypeError("The graph is not a tree")
        if root_node is None:
            root_node = list(nx.topological_sort(G))[0]

        def _hierarchy_pos(G, root, width=width, vert_gap=vert_gap, vert_loc=vert_loc,
                        xcenter=xcenter, pos=None):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                        vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos)
            return pos

        return _hierarchy_pos(G, root_node)        
    
    def build_graph_from_tree(self, tree):
        """
        Build a graph using networkx from the tree.
        """
        
        G = nx.DiGraph()
        
        def traverse(node, parent_id=None, edge_label=""):
            current_id = id(node)
            #Determine the node type and assign an appropriate label.
            if isinstance(node, Leaf):
                label = f"{node.label}"
                node_type = "leaf"
            else:
                label = f"{node.attribute_tested}"
                node_type = "node"
            G.add_node(current_id, label=label, type=node_type)
            if parent_id is not None:
                G.add_edge(parent_id, current_id, label=edge_label)
            if isinstance(node, Node):
                for value, child in node.children.items():
                    if child is not None:
                        traverse(child, current_id, edge_label=str(value))
        
        traverse(tree)
        return G

    def display_tree_graph(self):

        if self.id3_tree is None or self.id3_tree.tree is None:
            messagebox.showerror("Error", "No tree to display!")
            return

        # Building a graph from a tree
        G = self.build_graph_from_tree(self.id3_tree.tree)
        
        # Using the hierarchy_pos function to obtain a hierarchical layout
        root_id = id(self.id3_tree.tree)
        try:
            pos = self.hierarchy_pos(G, root_node=root_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute hierarchical layout: {str(e)}")
            return

        #Split nodes based on the "type" property
        leaf_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == "leaf"]
        non_leaf_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == "node"]

        plt.figure(figsize=(10, 8))
        
        nx.draw_networkx_nodes(G, pos, nodelist=non_leaf_nodes, node_color="skyblue", node_size=3500)
        nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_color="lightgreen", node_size=3500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        
        non_leaf_labels = {node: G.nodes[node]['label'] for node in non_leaf_nodes}
        nx.draw_networkx_labels(G, pos, labels=non_leaf_labels, font_size=15)
        
        for node in leaf_nodes:
            x, y = pos[node]
            label = G.nodes[node]['label']
            plt.text(x, y, label, fontsize=10, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center')
        
        plt.title("visualization of the Decision Tree")
        plt.axis('off')
        plt.show()



if __name__ == "__main__":
    app = DecisionTreeApp()
    app.mainloop()

