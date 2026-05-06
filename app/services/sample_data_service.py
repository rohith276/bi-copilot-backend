import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import uuid

BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"


def generate_sample_sales_data(row_count: int = 1000) -> str:
    """
    Generates a realistic sample sales dataset CSV file.
    """
    categories = ['Electronics', 'Home & Kitchen', 'Office Supplies', 'Apparel', 'Books']
    regions = ['North', 'South', 'East', 'West']
    products = {
        'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Monitor', 'Keyboard'],
        'Home & Kitchen': ['Blender', 'Coffee Maker', 'Toaster', 'Air Fryer', 'Mixer'],
        'Office Supplies': ['Chair', 'Desk', 'Printer', 'Paper', 'Markers'],
        'Apparel': ['T-Shirt', 'Jeans', 'Sweater', 'Jacket', 'Shoes'],
        'Books': ['Fiction', 'Non-Fiction', 'Biography', 'Textbook', 'Graphic Novel']
    }

    data = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(row_count):
        cat = np.random.choice(categories)
        prod = np.random.choice(products[cat])
        region = np.random.choice(regions)
        qty = np.random.randint(1, 10)
        unit_price = np.random.uniform(10, 500)
        tax_rate = 0.08
        discount_rate = np.random.choice([0, 0, 0, 0, 0.05, 0.1, 0.15])
        
        revenue = qty * unit_price
        discount_amt = revenue * discount_rate
        total_price = (revenue - discount_amt) * (1 + tax_rate)
        
        date = base_date + timedelta(days=np.random.randint(0, 365))
        
        data.append({
            'OrderDate': date.strftime('%Y-%m-%d'),
            'Region': region,
            'Category': cat,
            'Product': prod,
            'Quantity': qty,
            'UnitPrice': round(unit_price, 2),
            'Revenue': round(revenue, 2),
            'Discount': round(discount_amt, 2),
            'Total': round(total_price, 2),
            'CustomerRating': np.random.randint(1, 6),
            'StockLevel': np.random.randint(0, 100),
            'InventoryStatus': np.random.choice(['In Stock', 'In Stock', 'Low Stock', 'Out of Stock'])
        })


    df = pd.DataFrame(data)
    
    # Save to a temporary path
    filename = f"sample_sales_data_{uuid.uuid4().hex[:8]}.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename
    df.to_csv(file_path, index=False)
    
    return str(file_path)
