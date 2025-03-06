
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import numpy as np
import json
import csv

class DecisionTreeApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="cosmo")
        self.title("Decision Tree System")
        self.geometry("900x650")
        
        # Variables to store data
        self.dataset = None
        self.tree = None
        self.attribute_names = []
        
        # Configure notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the three main tabs
        self.create_data_input_tab()
        self.create_solution_tab()
        self.create_visualization_tab()
    
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
        
        # Button to load file
        load_button = ttk.Button(
            csv_frame, 
            text="Choose CSV File", 
            command=self.load_csv_file,
            bootstyle=(SUCCESS, OUTLINE)
        )
        load_button.pack(padx=10, pady=10)
        
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
    
    def create_solution_tab(self):
        """Create the solution tab (tree building)"""
        solution_tab = ttk.Frame(self.notebook)
        self.notebook.add(solution_tab, text="Build Tree")
        
        # Placeholder content
        ttk.Label(solution_tab, text="Tree building page will be implemented in part two").pack(pady=20)
    
    def create_visualization_tab(self):
        """Create the visualization tab"""
        viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(viz_tab, text="Display Tree & Predict")
        
        # Placeholder content
        ttk.Label(viz_tab, text="Tree visualization and prediction page will be implemented in part three").pack(pady=20)
    
    def load_csv_file(self):
        """Function to load a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
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

if __name__ == "__main__":
    app = DecisionTreeApp()
    app.mainloop()