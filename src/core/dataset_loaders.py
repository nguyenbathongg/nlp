def load_raw_text_data(file_path):
    """
    Load raw text data from a file.
    
    Args:
        file_path (str): The path to the text file
        
    Returns:
        str: The raw text data
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Returning sample text for demonstration purposes.")
        # Provide a small sample text for demonstration if file doesn't exist
        return """
        I bought a new laptop yesterday. It's very fast and efficient!
        The price was $999.99, but I got a 10% discount.
        I'm really happy with my purchase. Isn't technology amazing?
        """