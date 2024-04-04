package neural_plswork.connection.penalize;

public enum WeightPenalizer {
    CUSTOM(0),
    NONE(1),
    LASSO(2),
    RIDGE(3),
    ELASTIC(4),
    INVALID(-1);

    private final int id;
    private WeightPenalizer(int id) {
        this.id = id;
    }

    public WeightPenalizer getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return NONE;
            case 2:
                return LASSO;
            case 3:
                return RIDGE;
            case 4:
                return ELASTIC;
            default:
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}
