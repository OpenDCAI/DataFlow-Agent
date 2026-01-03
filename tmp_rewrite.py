
from my_module import EnhancedFileConverter

class Pipeline:
    def __init__(self):
        self.operators = {}
        self.register_operators()

    def register_operators(self):
        self.operators['EnhancedFileConverter'] = EnhancedFileConverter

    def run(self, data):
        # Assuming the first operator uses 'raw_content' as input_key
        input_key = 'raw_content'
        output_key = 'converted_content'

        # Process data through EnhancedFileConverter
        if 'EnhancedFileConverter' in self.operators:
            converter = self.operators['EnhancedFileConverter']()
            data[output_key] = converter.convert(data[input_key])
        else:
            raise Exception("Operator 'EnhancedFileConverter' not found in registry!")

        # Continue with other operations
        # Example: next_operator.process(data[output_key])

        return data

# Sample data
sample_data = [{'raw_content': 'Reading is a fundamental skill that not only improves vocabulary and comprehension but also stimulates imagination and critical thinking. Whether it is fiction, non-fiction, or poetry, literature provides insights into different cultures, histories, and perspectives.'}]

pipeline = Pipeline()
result = pipeline.run(sample_data[0])
print(result)
