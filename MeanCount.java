import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.NLineInputFormat;


public class MeanCount {
    private static double getMean(Text t) {
        return Double.parseDouble(t.toString().split("_")[0]);
    }

    private static int getCount(Text t) {
        return Integer.parseInt(t.toString().split("_")[1]);
    }

    private static Text wrapTuple(Double mean, Integer count) {
        return new Text(mean.toString() + "_" + count.toString());
    }

    public static Double aggregate(Double mean1, Integer count1, Double mean2, Integer count2) {
        return ((mean1 * count1) + (mean2 * count2)) / (count1 + count2);
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

        private final static IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] splitLine = value.toString().split(",");
            Text kraj = new Text(splitLine[5]);
            try {
                Double attendance = Double.parseDouble(splitLine[122]);
                context.write(kraj, wrapTuple(attendance, 1));
            } catch (NumberFormatException e) {
                return;
            }
        }
    }

    public static class IntMeanReducer extends Reducer<Text, Text, Text, Text> {

        @Override public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Double attendance = 0D;
            Integer totalCount = 0;
            for (Text t: values) {
                attendance = aggregate(attendance, totalCount, getMean(t), getCount(t));
                totalCount += getCount(t);
            }
            context.write(key, wrapTuple(attendance, totalCount));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "mean count");
        NLineInputFormat.setNumLinesPerSplit(job, 1);
        job.setJarByClass(MeanCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntMeanReducer.class);
        job.setReducerClass(IntMeanReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        Path output_path = new Path(args[1]);
        FileOutputFormat.setOutputPath(job, output_path);

        if (output_path.getFileSystem(conf).exists(output_path)) {
            output_path.getFileSystem(conf).delete(output_path, true);
        }

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}