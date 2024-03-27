package neural_plswork.initialize;

public class InvalidInitializationException extends RuntimeException {
    protected InvalidInitializationException(String message) {
        super(message);
    }

    protected InvalidInitializationException() {
        super("Invalid Initializer");
    }
}
