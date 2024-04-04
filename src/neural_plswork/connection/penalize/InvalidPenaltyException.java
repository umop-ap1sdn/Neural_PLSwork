package neural_plswork.connection.penalize;

public class InvalidPenaltyException extends RuntimeException {
    
    protected InvalidPenaltyException(String message) {
        super(message);
    }

    protected InvalidPenaltyException() {
        super("Invalid Penalty");
    }
}


