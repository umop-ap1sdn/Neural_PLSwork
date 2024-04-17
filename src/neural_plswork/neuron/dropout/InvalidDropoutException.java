package neural_plswork.neuron.dropout;

public class InvalidDropoutException extends RuntimeException {
    
    public InvalidDropoutException() {
        super("Invalid Dropout.");
    }

    public InvalidDropoutException(String message) {
        super(message);
    }
}
