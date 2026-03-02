import torch
import os

try:
    import torchviz
    HAS_TORCHVIZ = True
except ImportError:
    HAS_TORCHVIZ = False
    print("torchviz is not installed. Will only export ONNX model. Run 'pip install torchviz graphviz' to enable PDF/PNG generation.")

# Import the model from the existing script
from train_1dcnn import Text1DCNN

def main():
    VOCAB_SIZE = 25000
    EMBEDDING_DIM = 100
    NUM_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 1
    DROPOUT = 0.5

    print("Instantiating Text1DCNN model...")
    model = Text1DCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    )


    dummy_input = torch.randint(0, VOCAB_SIZE, (2, 10), dtype=torch.long)

    if HAS_TORCHVIZ:
        print("Generating architecture computation graph with torchviz...")
        # Forward pass to establish computation graph
        y = model(dummy_input)
        
        # Create and save graph
        graph = torchviz.make_dot(
            y, 
            params=dict(list(model.named_parameters()) + [('input', dummy_input)]), 
            show_attrs=True, 
            show_saved=True
        )
        
        output_path = "1dcnn_architecture"
        graph.render(output_path, format="pdf")
        graph.render(output_path, format="png")
        print(f"Saved computational graph to {output_path}.pdf and {output_path}.png")


    print("Exporting ONNX model for graphical viewers like Netron...")
    onnx_path = "1dcnn_architecture.onnx"
    # Exporting the model to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=14,          
        input_names=['text_tokens'],  
        output_names=['score_prediction'], 
        dynamic_axes={
            'text_tokens': {0: 'batch_size', 1: 'sequence_length'},
            'score_prediction': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported to {onnx_path}.")
    print("You can drop this .onnx file into https://netron.app/ to explore the 3D interactive graph of the model.")

if __name__ == "__main__":
    main()
