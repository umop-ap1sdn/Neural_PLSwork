package neural_plswork.neuron.activations;

public class InvalidActivationException extends RuntimeException {
    protected InvalidActivationException(String message) {
        super(message);
    }

    protected InvalidActivationException() {
        super("Invalid Activation Function");
    }
}
