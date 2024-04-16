package neural_plswork.dataframe.parsing;

public class ParsingException extends RuntimeException {
    protected ParsingException() {
        super("An error occurred during parsing of a dataframe");
    }
    
    protected ParsingException(String message) {
        super(message);
    }
}
