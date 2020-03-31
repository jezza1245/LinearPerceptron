package com.company;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.util.Iterator;
import java.util.Random;

public class WekaTools {

    public static File dataCollection;
    static{
        dataCollection = new File("resources\\UCIContinuous");
    }

    public static Instances loadClassificationData(String filePath){
        Instances data;
        try{
            FileReader reader = new FileReader(filePath);
            data = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
            data = null;
        }
        return data;
    }

    public static Instances getDataSet(String datasetName) throws Exception{

        File folder = dataCollection.listFiles((dir, name) -> name.contains(datasetName))[0];
        if(folder == null) throw new Exception();

        File data = folder.listFiles((_folder,_name) -> !_name.endsWith("TRAIN.arff") && !_name.endsWith("TEST.arff"))[0];

        return loadClassificationData(data.getPath());
    }

    public static Instances[] getDataSetSplit(String datasetName) throws Exception{

        File folder = dataCollection.listFiles((dir, name) -> name.contains(datasetName))[0];
        if(folder == null) throw new Exception();

        File train = folder.listFiles((_folder,_name) -> _name.endsWith("TRAIN.arff"))[0];
        File test = folder.listFiles((_folder,_name) -> _name.endsWith("TEST.arff"))[0];

        Instances split[] = new Instances[]{loadClassificationData(train.getPath()),loadClassificationData(test.getPath())};
        return split;

    }

    public splitDatasetIterator getSplitDatasetIterator(){
        return new splitDatasetIterator();
    }

    private class splitDatasetIterator implements Iterator{

        File datasets[];
        int index = 0;

        public splitDatasetIterator(){
            datasets = dataCollection.listFiles();
        }

        @Override
        public boolean hasNext() {
            return index < (datasets.length);
        }

        @Override
        public Object next() {
            File dataset = datasets[index++];
            if(!dataset.isDirectory()) dataset = datasets[index++];
            Instances split[] = new Instances[2];
            split[0] = loadClassificationData(dataset.listFiles((dir, name) -> name.endsWith("TRAIN.arff"))[0].getPath());
            split[1] = loadClassificationData(dataset.listFiles((dir, name) -> name.endsWith("TEST.arff"))[0].getPath());
            return split;
        }

    }

    public datasetIterator getDatasetIterator(){
        return new datasetIterator();
    }

    private class datasetIterator implements Iterator{

        File datasets[];
        int index = 0;

        public datasetIterator(){
            datasets = dataCollection.listFiles();
        }

        @Override
        public boolean hasNext() {
            return index < (datasets.length);
        }

        @Override
        public Object next() {
            File dataset = datasets[index++];
            if(!dataset.isDirectory()) dataset = datasets[index++];
            Instances data = loadClassificationData(dataset.listFiles((dir, name) -> !name.endsWith("TRAIN.arff") && !name.endsWith("TEST.arff"))[0].getPath());
            return data;
        }

    }

    public static double[] evaluationMetrics(Classifier classifier, Instances instances, int folds) throws Exception{

        double metrics[] = new double[6];

        Evaluation crossValidate = new Evaluation(instances);
        crossValidate.crossValidateModel(classifier, instances, folds, new java.util.Random(1));

        metrics[0] = crossValidate.correct() / crossValidate.numInstances(); // accuracy
        //metrics[1] = crossValidate.errorRate(); // CV Error Rate
        metrics[1] = crossValidate.areaUnderROC(0); // AUROC
        //metrics[3] = crossValidate.areaUnderROC(instances.classIndex()); //NLL
        //metrics[4] = crossValidate.areaUnderROC(instances.classIndex()); // Sensitivity
        //metrics[5] = crossValidate.areaUnderROC(instances.classIndex()); // Specificity

        return metrics;
    }

    public static double crossValError(Classifier classifier, Instances instances, int folds) throws Exception{

        Evaluation crossValidate = new Evaluation(instances);
        crossValidate.crossValidateModel(classifier, instances, folds, new java.util.Random(1));

        return crossValidate.errorRate();
    }

    public static Instances[] splitData(Instances all, double proportion){

        Instances split[] = new Instances[2];
        int totalInstances = all.numInstances();

        int splitAt = (int) (proportion*totalInstances);

        all.randomize(new Random(42));
        split[0] = new Instances(all, 0, splitAt);
        split[1] = new Instances(all, splitAt, totalInstances-splitAt);

        return split;
    }

    public static double accuracy(Classifier classifier, Instances test) {
        int totalInstances = test.numInstances();
        int numCorrect = 0;

        Attribute classAttribute = test.classAttribute();
        for (Instance instance : test) {
            try {
                double predicted = classifier.classifyInstance(instance);
                double actual = instance.value(classAttribute);
                if(predicted==-1) predicted = 0;
                if(actual==-1) actual = 0;

                if (predicted == actual) {
                    numCorrect++;
                }
            } catch (Exception e) {
                System.out.println("Error classifying instance");

            }
        }

        double acc = numCorrect/(double)totalInstances;

        return acc;
    }

    public static void main(String[] args) {
        WekaTools wk = new WekaTools();
        Iterator dsit = wk.getDatasetIterator();
        while(dsit.hasNext()){
            Instances ds = (Instances)dsit.next();
            ds.setClassIndex(ds.numAttributes() - 1);
            System.out.print(ds.relationName() + "  -> ");
            int class0 = 0, class1 = 0;
            for(Instance i : ds){
                if(i.classValue() == 0.0){
                    class0++;
                }else{
                    class1++;
                }
            }
            System.out.print(class0 + "  " + class1 + "\n\n");

        }
    }

}
