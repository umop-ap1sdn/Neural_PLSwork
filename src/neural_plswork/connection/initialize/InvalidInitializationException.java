package neural_plswork.connection.initialize;

public class InvalidInitializationException extends RuntimeException {
    protected InvalidInitializationException(String message) {
        super(message);
    }

    protected InvalidInitializationException() {
        super("Invalid Initializer");
    }
}
