package com.lanedetector;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends Activity {
    private LaneDetectorClient client;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        client = new LaneDetectorClient(this);

    }

    @Override
    protected void onResume() {
        super.onResume();

        Log.e("123", "--> " + this.getApplicationInfo().nativeLibraryDir);
        File directory = new File(this.getApplicationInfo().nativeLibraryDir);
        File[] files = directory.listFiles();
        Log.d("Files", "Size: "+ files.length);
        for (int i = 0; i < files.length; i++)
        {
            Log.d("Files", "FileName:" + files[i].getName());
        }

        AssetManager assetManager = this.getAssets();
        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open("20.jpg");
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }
        client.detect(bitmap);
    }


}