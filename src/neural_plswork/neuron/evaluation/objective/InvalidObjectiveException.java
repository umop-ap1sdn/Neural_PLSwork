package neural_plswork.neuron.evaluation.objective;

public class InvalidObjectiveException extends RuntimeException {
    public InvalidObjectiveException() {
        super("Invalid objective function");
    }

    public InvalidObjectiveException(String message) {
        super(message);
    }
}
