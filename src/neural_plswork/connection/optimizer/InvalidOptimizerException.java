package neural_plswork.connection.optimizer;

public class InvalidOptimizerException extends RuntimeException {
    protected InvalidOptimizerException(String message) {
        super(message);
    }

    protected InvalidOptimizerException() {
        super("Invalid Optimizer");
    }
}
