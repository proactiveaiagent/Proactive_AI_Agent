package com.screenrecorder.mvp

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ScreenRecordService : Service() {

    private var mediaProjection: MediaProjection? = null
    private var mediaRecorder: MediaRecorder? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var outputPath: String? = null
    private var isRecording = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1)
                val resultData = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_RESULT_DATA)
                }
                if (resultData != null) {
                    startForeground(NOTIFICATION_ID, createNotification())
                    try {
                        startRecording(resultCode, resultData)
                    } catch (e: Exception) {
                        Log.e(TAG, "startRecording failed", e)
                        sendError("录制启动失败: ${e.message}")
                        stopForeground(STOP_FOREGROUND_REMOVE)
                        stopSelf()
                    }
                } else {
                    Log.e(TAG, "resultData is null, cannot start recording")
                    sendError("录制启动失败: 未获取到屏幕录制权限数据")
                    stopSelf()
                }
            }
            ACTION_STOP -> {
                stopRecording()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun startRecording(resultCode: Int, resultData: Intent) {
        val projectionManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = projectionManager.getMediaProjection(resultCode, resultData)
        if (mediaProjection == null) {
            Log.e(TAG, "getMediaProjection returned null (resultCode=$resultCode)")
            sendError("录制启动失败: MediaProjection 为 null，可能权限被拒绝")
            return
        }
        Log.d(TAG, "MediaProjection obtained successfully")

        mediaProjection!!.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                Log.d(TAG, "MediaProjection.Callback.onStop()")
                isRecording = false
            }
        }, null)

        val wm = getSystemService(Context.WINDOW_SERVICE) as android.view.WindowManager
        val displayMetrics = resources.displayMetrics
        val width = displayMetrics.widthPixels
        val height = displayMetrics.heightPixels
        val density = displayMetrics.densityDpi
        Log.d(TAG, "Screen: ${width}x${height} density=$density")

        val dir = File(
            getExternalFilesDir(Environment.DIRECTORY_MOVIES),
            "ScreenRecords"
        ).apply { mkdirs() }
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val file = File(dir, "record_${dateFormat.format(Date())}.mp4")
        outputPath = file.absolutePath
        Log.d(TAG, "Output file: ${file.absolutePath}")

        mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(this)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }

        mediaRecorder!!.apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoSize(width, height)
            setVideoFrameRate(30)
            setVideoEncodingBitRate(6_000_000)
            setOutputFile(file.absolutePath)
            try {
                prepare()
                Log.d(TAG, "MediaRecorder prepared")
            } catch (e: Exception) {
                Log.e(TAG, "MediaRecorder.prepare() failed", e)
                sendError("录制准备失败: ${e.message}")
                release()
                mediaRecorder = null
                return
            }
        }

        virtualDisplay = mediaProjection!!.createVirtualDisplay(
            "XOOGUIAGT",
            width, height, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            mediaRecorder!!.surface,
            null,
            null
        )
        Log.d(TAG, "VirtualDisplay created")

        mediaRecorder!!.start()
        isRecording = true
        Log.d(TAG, "MediaRecorder started - recording is active")
    }

    private fun stopRecording() {
        Log.d(TAG, "stopRecording() called, isRecording=$isRecording")
        try {
            if (isRecording) {
                mediaRecorder?.stop()
                Log.d(TAG, "MediaRecorder stopped")
            }
        } catch (e: Exception) {
            Log.e(TAG, "MediaRecorder.stop() error", e)
        }
        try {
            mediaRecorder?.release()
        } catch (e: Exception) {
            Log.e(TAG, "MediaRecorder.release() error", e)
        }
        mediaRecorder = null

        try {
            virtualDisplay?.release()
        } catch (e: Exception) {
            Log.e(TAG, "VirtualDisplay.release() error", e)
        }
        virtualDisplay = null

        try {
            mediaProjection?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "MediaProjection.stop() error", e)
        }
        mediaProjection = null

        isRecording = false
        val fileSize = outputPath?.let { File(it).length() } ?: 0
        Log.d(TAG, "Recording stopped. File: $outputPath, size: $fileSize bytes")
        if (fileSize == 0L) {
            Log.w(TAG, "WARNING: recorded file is 0 bytes!")
        }
    }

    private fun sendError(message: String) {
        val intent = Intent(BROADCAST_ERROR).apply {
            putExtra("error_msg", message)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun createNotification(): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.app_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply { setShowBadge(false) }
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.recording_notification))
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val TAG = "ScreenRecordSvc"
        const val ACTION_START = "com.screenrecorder.mvp.START"
        const val ACTION_STOP = "com.screenrecorder.mvp.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val BROADCAST_ERROR = "com.screenrecorder.mvp.RECORD_ERROR"
        private const val CHANNEL_ID = "screen_record"
        private const val NOTIFICATION_ID = 1001
    }
}
