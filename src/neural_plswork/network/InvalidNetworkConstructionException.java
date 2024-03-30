package neural_plswork.network;

public class InvalidNetworkConstructionException extends RuntimeException {
    public InvalidNetworkConstructionException() {
        super("Invalid Network Construction Exception");
    }

    public InvalidNetworkConstructionException(String message) {
        super(message);
    }
}
