package com.lanedetector;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class LaneDetectorClient {
    private final String TAG="LaneDetectorClient";
    private final String MODEL_NAME="model_int8.tflite";
    private Interpreter tfLite = null;
    private HexagonDelegate hexagonDelegate = null;
    private TensorImage inBuffer;
    private TensorBuffer outBuffer;
    private int inWidth;
    private int inHeight;

    public LaneDetectorClient() {

    }

    public LaneDetectorClient(Context contex) {
        init(contex);
    }

    public void init(Context context) {
        Interpreter.Options options = (new Interpreter.Options());

        /// Create the Delegate instance.
        try {
            hexagonDelegate = new HexagonDelegate(context);
            options.addDelegate(hexagonDelegate);
        } catch (UnsupportedOperationException e) {
            // Hexagon delegate is not supported on this device.
            throw new IllegalStateException("Fail to create Hexagon delegate");
        }

        // Initialize TFLite interpreter
        MappedByteBuffer buf = null;
        try {
            buf = loadModelFile(context.getAssets(), MODEL_NAME);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        try {
            tfLite = new Interpreter(buf, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        // create the input tensor
        int inTensorIndex = 0;
        int[] inShape = tfLite.getInputTensor(inTensorIndex).shape(); // {1, height, width, 3}
        DataType inDataType = tfLite.getInputTensor(inTensorIndex).dataType();
        inBuffer = new TensorImage(inDataType);
        inWidth = inShape[2];
        inHeight = inShape[1];

        // create the output tensor
        int outTensorIndex = 0;
        int[] outShape = tfLite.getOutputTensor(outTensorIndex).shape(); // {1, max_lane_count, y_anchors, x_anchors}
        DataType outDataType = tfLite.getOutputTensor(outTensorIndex).dataType();
        outBuffer = TensorBuffer.createFixedSize(outShape, outDataType);

        Log.i(TAG, "Input tensor shape  : [" + inShape[0] + ", " + inShape[1] + "," + inShape[2] + "," + inShape[3] + "], dtype " + inDataType.toString() );
        Log.i(TAG, "Output tensor shape : [" + outShape[0] + ", " + outShape[1] + "," + outShape[2] + "," + outShape[3] + "], dtype " + outDataType.toString() );
    }

    public void release() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = null;
        }

        if (hexagonDelegate != null) {
            hexagonDelegate.close();
            hexagonDelegate = null;
        }
    }

    public void detect(final Bitmap bitmap) {
        if (tfLite == null) {
            throw new IllegalStateException("TF-Lite not init yet or init failed.");
        }

        inBuffer = loadImage(bitmap);
        tfLite.run(inBuffer.getBuffer(), outBuffer.getBuffer());

//        startTimeForLoadImage = SystemClock.uptimeMillis();
//        for (int i = 0 ; i < 100 ; i++) {
//            tfLite.run(inBuffer.getBuffer(), outBuffer.getBuffer());
//        }
//        endTimeForLoadImage = SystemClock.uptimeMillis();
//        Log.d(TAG, "avg Timecost to inference: " + (endTimeForLoadImage - startTimeForLoadImage) / 100.0);
//        Log.d(TAG, "output array " + outBuffer.getIntArray().length);


        byte[] bitmapdata = new byte[4 * 72 * 128];
        Arrays.fill(bitmapdata, (byte)0);
        int intAry[] = outBuffer.getIntArray();
        for (int laneIdx = 0 ; laneIdx < 4 ; laneIdx++) {

            for (int dy = 0; dy < 72; dy++) {
                int max = 0;
                int max_dx = 0;

                for (int dx = 0; dx < 128; dx++) {
                    int idx = laneIdx * 128 * 72 + dy * 128 + dx;
                    if (intAry[idx] > max) {
                        max = intAry[idx];
                        max_dx = dx;
                    }
                }

                bitmapdata[dy * 128*4 + max_dx * 4 +0] = (byte)255;
                bitmapdata[dy * 128*4 + max_dx * 4 +1] = (byte)0;
                bitmapdata[dy * 128*4 + max_dx * 4 +2] = (byte)0;
                bitmapdata[dy * 128*4 + max_dx * 4 +3] = (byte)255;
            }
        }
        Bitmap result = Bitmap.createBitmap(128, 72, Bitmap.Config.ARGB_8888);
        ByteBuffer buffer = ByteBuffer.wrap(bitmapdata);
        result.copyPixelsFromBuffer(buffer);

        try (FileOutputStream out = new FileOutputStream("/sdcard/a.png")) {
            result.compress(Bitmap.CompressFormat.PNG, 100, out);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inBuffer.load(bitmap);

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .build();
        return imageProcessor.process(inBuffer);
    }

    /** Load TF Lite model from assets. */
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String model_name) throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(model_name);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }
}
