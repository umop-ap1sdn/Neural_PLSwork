package neural_plswork.dataframe.parsing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;

import neural_plswork.dataframe.Column;
import neural_plswork.dataframe.DataFrame;

public class DataFrameParser {

    private final String[] DEFAULT_NULL_STRINGS = {"", "null", "NA", "N/A", null};
    private final HashSet<String> KNOWN_NULL_STRINGS;
    
    public DataFrameParser() {
        this.KNOWN_NULL_STRINGS = new HashSet<>();
        for(String s: DEFAULT_NULL_STRINGS) KNOWN_NULL_STRINGS.add(s);
    }

    public DataFrameParser(String[] null_strings, boolean append) {
        this.KNOWN_NULL_STRINGS = new HashSet<>();
        if(append) {
            for(String s: DEFAULT_NULL_STRINGS) KNOWN_NULL_STRINGS.add(s);
        }

        for(String s: null_strings) KNOWN_NULL_STRINGS.add(s);
    }

    public DataFrame parse(String label, String[] data, boolean allowNull) {
        String[] labels = label.split(",");
        String[][] rows = new String[data.length][];

        for(int i = 0; i < rows.length; i++) {
            rows[i] = data[i].split(",");
        }

        String[][] columnsArr = transposeColumns(rows);
        Column<?>[] columns = new Column[columnsArr.length];
        
        for(int i = 0; i < columns.length; i++) {
            columns[i] = ColumnConverter.convert(new Column<>(columnsArr[i]), KNOWN_NULL_STRINGS, allowNull);
        }
        
        return new DataFrame(labels, columns);
    }

    private DataFrame read(BufferedReader br, boolean allowNull) throws ParsingException {
        try {
        String labels = br.readLine();
        String line = br.readLine();

        ArrayList<String> read = new ArrayList<>();
        while(line != null) {
            read.add(line);
            line = br.readLine();
        }

        String[] data = new String[read.size()];
        read.toArray(data);
        return parse(labels, data, allowNull);

        } catch (IOException e) {
            throw new ParsingException();
        }
    }

    public DataFrame read_string(String dataframe, boolean allowNull) throws ParsingException {
        BufferedReader br = new BufferedReader(new StringReader(dataframe));
        return read(br, allowNull);
    }

    public DataFrame read_csv(String file, boolean allowNull) throws ParsingException {
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(file)));
            return read(br, allowNull);
        } catch (FileNotFoundException e) {
            throw new ParsingException("Input file not found");
        }
    }

    private String[][] transposeColumns(String[][] input) {
        String[][] output = new String[input[0].length][input.length];

        for(int i = 0; i < input.length; i++) {
            for(int j = 0; j < input[i].length; j++) {
                output[j][i] = input[i][j];
            }
        }

        return output;
    }
}
