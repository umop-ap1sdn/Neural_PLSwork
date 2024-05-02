package neural_plswork.dataframe.parsing;

import java.util.HashSet;

import neural_plswork.dataframe.Column;

public class ColumnConverter {
    
    private static final int STRING_ELIGIBLE = -1;
    private static final int BOOL_ELIGIBLE = 0;
    private static final int INT_ELIGIBLE = 1;
    private static final int DOUBLE_ELIGIBLE = 2;

    protected static Column<?> convert(Column<String> column, HashSet<String> null_strings, boolean allowNull) {
        int eligibility = determine_eligibility(column, null_strings);

        switch(eligibility) {
            case BOOL_ELIGIBLE: return convertToBool(column, null_strings, allowNull);
            case INT_ELIGIBLE: return convertToInt(column, null_strings, allowNull);
            case DOUBLE_ELIGIBLE: return convertToDouble(column, null_strings, allowNull);
            default: return column;
        }
    }

    private static Column<Boolean> convertToBool(Column<String> column, HashSet<String> null_strings, boolean allowNull) {
        Boolean[] data = new Boolean[column.size()];

        for(int i = 0; i < column.size(); i++) {
            if(null_strings.contains(column.get(i))) {
                if(allowNull) data[i] = null;
                else data[i] = false;
                continue;
            }

            data[i] = Boolean.parseBoolean(column.get(i));
        }

        return new Column<>(data);
    }

    private static Column<Integer> convertToInt(Column<String> column, HashSet<String> null_strings, boolean allowNull) {
        Integer[] data = new Integer[column.size()];

        for(int i = 0; i < column.size(); i++) {
            if(null_strings.contains(column.get(i))) {
                if(allowNull) data[i] = null;
                else data[i] = 0;
                continue;
            }

            data[i] = Integer.parseInt(column.get(i));
        }

        return new Column<>(data);
    }

    private static Column<Double> convertToDouble(Column<String> column, HashSet<String> null_strings, boolean allowNull) {
        Double[] data = new Double[column.size()];

        for(int i = 0; i < column.size(); i++) {
            if(null_strings.contains(column.get(i))) {
                if(allowNull) data[i] = null;
                else data[i] = 0.0;
                continue;
            }

            data[i] = Double.parseDouble(column.get(i));
        }

        return new Column<>(data);
    }

    private static int determine_eligibility(Column<String> column, HashSet<String> null_strings) {
        boolean boolEligible = true;
        boolean intEligible = true;
        boolean doubleEligible = true;

        for(String s: column) {
            if(null_strings.contains(s)) continue;
            if(!boolEligible && !intEligible && !doubleEligible) return STRING_ELIGIBLE;

            if(s.equals("false") || s.equals("true")) {
                intEligible = false;
                doubleEligible = false;
            } else {
                boolEligible = false;
            }

            try {
                Integer.parseInt(s);
            } catch (NumberFormatException e) {
                intEligible = false;
            }

            try {
                Double.parseDouble(s);
            } catch (NumberFormatException e) {
                doubleEligible = false;
            }
        }

        if(boolEligible) return BOOL_ELIGIBLE;
        if(intEligible) return INT_ELIGIBLE;
        if(doubleEligible) return DOUBLE_ELIGIBLE;
        
        return STRING_ELIGIBLE;
    }
}
