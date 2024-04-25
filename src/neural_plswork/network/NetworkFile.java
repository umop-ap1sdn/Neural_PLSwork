package neural_plswork.network;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class NetworkFile {
    public static boolean writeFile(Network nn, int resolution, String path, String name, boolean overwrite) {
        File file = new File(path);
        if(!file.exists()) file.mkdirs();

        int extPoint = name.indexOf(".");
        String fileName = name;
        String extension = ".nn";
        if(extPoint != -1) {
            fileName = name.substring(0, extPoint);
            extension = name.substring(extPoint);
        }

        File writeTo;

        if(!overwrite) {
            int id = 0;
            File check;
            do {
                String trueName = fileName + id + extension;
                check = new File(path + File.separator + trueName);
                id++;
            } while (check.exists());

            writeTo = check;
        } else {
            writeTo = new File(path + File.separator + fileName + extension);
        }

        return writeFile(nn, writeTo, resolution);
    }

    private static boolean writeFile(Network nn, File writeTo, int resolution) {
        try (FileWriter fw = new FileWriter(writeTo, false)) {
            String[] networkString = nn.networkToString(resolution);
            for(String s: networkString) {
                fw.write(s);
                fw.flush();
            }
            
            fw.close();

        } catch (IOException e) {
            return false;
        }

        return true;
    }
}
