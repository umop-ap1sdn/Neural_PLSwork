package neural_plswork.evaluation.loss;

public class InvalidLossException extends RuntimeException {
    protected InvalidLossException(String message) {
        super(message);
    }

    protected InvalidLossException() {
        super("Invalid Loss Function");
    }
}
