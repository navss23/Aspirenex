package com.example.handwriting.controller;

import com.example.handwriting.service.HandwritingService;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;

@RestController
@RequestMapping("/api/handwriting")
public class HandwritingController {

    @Autowired
    private HandwritingService handwritingService;

    @PostMapping("/train")
    public String trainModel(@RequestParam("imagesDir") MultipartFile[] imagesDir, 
                             @RequestParam("csvFile") MultipartFile csvFile,
                             @RequestParam int batchSize, @RequestParam int lstmLayerSize, 
                             @RequestParam int tbpttLength, @RequestParam double learningRate, 
                             @RequestParam int numEpochs) {
        try {
            // Save the uploaded files to a temporary directory
            File tempDir = new File(System.getProperty("java.io.tmpdir"), "handwriting");
            if (!tempDir.exists()) {
                tempDir.mkdirs();
            }

            File imagesDirPath = new File(tempDir, "images");
            if (!imagesDirPath.exists()) {
                imagesDirPath.mkdirs();
            }

            for (MultipartFile imageFile : imagesDir) {
                File destFile = new File(imagesDirPath, imageFile.getOriginalFilename());
                imageFile.transferTo(destFile);
            }

            File csvFilePath = new File(tempDir, csvFile.getOriginalFilename());
            csvFile.transferTo(csvFilePath);

            // Train the model
            handwritingService.trainModel(imagesDirPath.getAbsolutePath(), csvFilePath.getAbsolutePath(), batchSize, lstmLayerSize, tbpttLength, learningRate, numEpochs);
            return "Model trained successfully!";
        } catch (Exception e) {
            e.printStackTrace();
            return "Error during training: " + e.getMessage();
        }
    }

    @GetMapping("/generate")
    public String generateText(@RequestParam int length) {
        try {
            File modelFile = new File("handwriting-model.zip");
            if (!modelFile.exists()) {
                return "Model file not found. Please train the model first.";
            }

            MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
            return handwritingService.generateText(model, length);
        } catch (Exception e) {
            e.printStackTrace();
            return "Error during text generation: " + e.getMessage();
        }
    }
}
