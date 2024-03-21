package neural_plswork.activations;

public class InvalidActivationException extends RuntimeException {
    protected InvalidActivationException(String message) {
        super(message);
    }

    protected InvalidActivationException() {
        super("Invalid Activation Function");
    }
}
