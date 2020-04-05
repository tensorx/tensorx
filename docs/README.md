## Documentation Configuration

1. Document methods for a class in a source file:
```json
 {
        'page': 'folder/page_name.md',
        'source': 'filepath.py',
        'methods': [
            ('ClassName', 'method_name'),
            # all methods from a class
            ('ClassName', '*')
        ]
 }
```    
**page**: path to markdown template relative to templates folder

   

2. Document entire classes from a given source file:
    {
        'page': 'folder/page_name.md',
        'source': 'filepath.py',
        'classes': [
            'ClassName1',
            'ClassName2'
        ]
    }

3. Document functions from a source file:
     {
        'page': 'folder/page_name.md',
        'source': 'filepath.py',
        'functions': [
            'function_name1',
            'function_name2'
        ]
    }



"""