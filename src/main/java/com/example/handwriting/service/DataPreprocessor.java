package com.example.handwriting.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataPreprocessor {

    private static final int IMG_WIDTH = 28; // Width of resized images
    private static final int IMG_HEIGHT = 28; // Height of resized images
    private static final int SEQ_LENGTH = 100;

    public ListDataSetIterator<DataSet> getDataSetIterator(String imagesDirPath, String csvFilePath, int batchSize) throws Exception {
        List<String> labels = readLabels(csvFilePath);
        List<DataSet> dataSetList = new ArrayList<>();

        File imagesDir = new File(imagesDirPath);
        File[] imageFiles = imagesDir.listFiles();

        if (imageFiles == null) {
            throw new IllegalArgumentException("Invalid images directory path.");
        }

        for (File imageFile : imageFiles) {
            BufferedImage image = ImageIO.read(imageFile);
            BufferedImage resizedImage = resizeImage(image, IMG_WIDTH, IMG_HEIGHT);
            INDArray input = imageToINDArray(resizedImage);
            String label = findLabel(imageFile.getName(), labels);
            if (label != null) {
                INDArray output = labelToINDArray(label);
                dataSetList.add(new DataSet(input, output));
            }
        }

        return new ListDataSetIterator<>(dataSetList, batchSize);
    }

    private List<String> readLabels(String csvFilePath) throws Exception {
        FileReader reader = new FileReader(csvFilePath);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
        List<String> labels = new ArrayList<>();
        for (CSVRecord record : records) {
            labels.add(record.get("name")); // Adjust column name as per your CSV file
        }
        return labels;
    }

    private String findLabel(String imageName, List<String> labels) {
        // Implement logic to find the corresponding label for the image
        // This is a placeholder and needs to be implemented based on your CSV structure
        return labels.stream().filter(label -> imageName.contains(label)).findFirst().orElse(null);
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        resizedImage.getGraphics().drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        return resizedImage;
    }

    private INDArray imageToINDArray(BufferedImage image) {
        int[] shape = {1, IMG_WIDTH * IMG_HEIGHT};
        INDArray indArray = Nd4j.create(shape);
        int index = 0;
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF; // Extract grayscale value
                indArray.putScalar(index++, gray / 255.0); // Normalize to [0, 1]
            }
        }
        return indArray;
    }

    private INDArray labelToINDArray(String label) {
        INDArray indArray = Nd4j.zeros(1, label.length());
        for (int i = 0; i < label.length(); i++) {
            indArray.putScalar(new int[]{0, i}, label.charAt(i));
        }
        return indArray;
    }

    public MultiLayerNetwork getNetwork(int lstmLayerSize, int tbpttLength, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new LSTM.Builder().nIn(IMG_WIDTH * IMG_HEIGHT).nOut(lstmLayerSize).activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(lstmLayerSize).nOut(SEQ_LENGTH).build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(tbpttLength)
                .tBPTTBackwardLength(tbpttLength)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }
}
