from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime
import fitz  
import base64  

# Load environment variables
load_dotenv()

class SupabaseHandler:
    def __init__(self):
        # Load Supabase credentials from environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
        
        # Initialize Supabase client
        self.client = create_client(supabase_url, supabase_key)

    def insert_document(self, filename, extension, file_type, size_kb, author, description, num_pages):
        """Insert a document record into the 'documents' table."""
        response = self.client.table('documents').insert({
            "filename": filename,
            "extension": extension,
            "file_type": file_type,
            "size_kb": int(size_kb),
            "author": author,
            "description": description,
            "num_pages": num_pages,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
        
        # Return the document_id of the newly inserted record
        return response.data[0]['document_id']

    def insert_page(self, document_id, page_number, image_path):
        """Insert a page record into the 'document_pages' table."""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Encode image data to Base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        response = self.client.table('document_pages').insert({
            "document_id": document_id,
            "page_number": page_number,
            "image": image_base64,  # Use the Base64 encoded image
            "created_at": datetime.now().isoformat()
        }).execute()
        
        return response.data[0]['page_id']

# Function to convert PDF to JPG and store in database
def process_document(file_path, author, description):
    handler = SupabaseHandler()
    
    # Extract file details
    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1]
    file_type = "PDF"  # Assuming PDF for this example
    size_kb = os.path.getsize(file_path) / 1024
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    # Open the PDF
    doc = fitz.open(file_path)
    num_pages = doc.page_count
    
    # Insert document record
    document_id = handler.insert_document(filename, extension, file_type, size_kb, author, description, num_pages)
    
    # Convert each page to JPG and insert into document_pages table
    for page_num in range(num_pages):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = f"{output_dir}/{filename}-p{page_num + 1}.jpg"
        pix.save(image_path)
        
        handler.insert_page(document_id, page_num + 1, image_path)
    
    print(f"Document processed and stored in database. Document ID: {document_id}")

# Main function
if __name__ == "__main__":
    file_path = input("Enter the path to the document: ")
    author = input("Enter the author: ")
    description = input("Enter a description: ")
    
    process_document(file_path, author, description)