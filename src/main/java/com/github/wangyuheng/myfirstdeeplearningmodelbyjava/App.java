package com.github.wangyuheng.myfirstdeeplearningmodelbyjava;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import ai.djl.util.ZipUtils;
import com.github.wangyuheng.myfirstdeeplearningmodelbyjava.mnist.MnistInference;
import com.github.wangyuheng.myfirstdeeplearningmodelbyjava.mnist.MnistTrainer;
import com.github.wangyuheng.myfirstdeeplearningmodelbyjava.util.Arguments;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

/**
 * @author wangyuheng
 */
public final class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) throws IOException, TranslateException {
//        initDataset();
//        train();
        inference();
    }

    private static void initDataset() throws IOException {
        if (!Paths.get("dataset", "mnist").toFile().exists()) {
            ZipUtils.unzip(Files.newInputStream(Paths.get("dataset", "mnist.zip")), Paths.get("dataset"));
        }
    }

    private static void train() throws IOException, TranslateException {
        Arguments arguments = new Arguments();
        arguments.setEpoch(1);
        MnistTrainer.getInstance().train(arguments);
    }

    private static void inference() throws IOException {
        try (Stream<Path> paths = Files.list(Paths.get("src", "main", "resources", "mnist"))) {
            paths.filter(p -> p.toString().endsWith(".png"))
                    .forEach(p -> {
                        try {
                            Integer res = MnistInference.recognition(ImageFactory.getInstance().fromFile(p));
                            log.info("inference file:{} result:{}", p, res);
                        } catch (IOException | TranslateException | MalformedModelException e) {
                            e.printStackTrace();
                        }
                    });
        }

    }

}
