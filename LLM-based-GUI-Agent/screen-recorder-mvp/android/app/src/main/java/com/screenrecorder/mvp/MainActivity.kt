package com.screenrecorder.mvp

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.DrawableRes
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.core.content.ContextCompat
import androidx.core.text.HtmlCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private lateinit var btnPcRecord: MaterialButton
    private lateinit var btnPhoneRecord: MaterialButton
    private lateinit var tvPcRecordStatus: TextView
    private lateinit var tvRecordStatus: TextView
    private lateinit var tvRecordPath: TextView
    private lateinit var etPcIp: TextInputEditText
    private lateinit var etPcPort: TextInputEditText
    private lateinit var spinnerVideos: Spinner
    private lateinit var btnUpload: MaterialButton
    private lateinit var tvUploadStatus: TextView
    private lateinit var bottomNav: BottomNavigationView
    private lateinit var sectionRecording: ScrollView
    private lateinit var sectionOfflineRecording: ScrollView
    private lateinit var sectionEarnings: View
    private lateinit var sectionData: View
    private lateinit var sectionSettings: View

    private var isPhoneRecording = false
    private var isPcRecording = false
    private var earningsSurveyExpanded = true
    private var earningsRecExpanded = false
    private var earningsDiaryExpanded = false
    private var recordingFiles: List<File> = emptyList()
    private var selectedUploadFile: File? = null

    private val errorReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val msg = intent?.getStringExtra("error_msg") ?: "Unknown error"
            Log.e(TAG, "Received error from service: $msg")
            isPhoneRecording = false
            updatePhoneRecordButtonUi()
            tvRecordStatus.text = "Phone: recording failed"
            Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
        }
    }

    private val projectionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        Log.d(TAG, "projectionLauncher callback: resultCode=${result.resultCode}, data=${result.data}")
        if (result.resultCode == RESULT_OK && result.data != null) {
            startRecordService(result.resultCode, result.data!!)
        } else {
            isPhoneRecording = false
            updatePhoneRecordButtonUi()
            tvRecordStatus.text = "Phone: not recording (permission denied)"
            Toast.makeText(
                this,
                "Screen capture permission is required. Tap Start and choose \"Start now\" in the system dialog.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (!granted && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Toast.makeText(this, "Notification permission helps keep recording visible in the background.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        supportActionBar?.hide()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        btnPcRecord = findViewById(R.id.btnPcRecord)
        btnPhoneRecord = findViewById(R.id.btnPhoneRecord)
        tvPcRecordStatus = findViewById(R.id.tvPcRecordStatus)
        tvRecordStatus = findViewById(R.id.tvRecordStatus)
        tvRecordPath = findViewById(R.id.tvRecordPath)
        etPcIp = findViewById(R.id.etPcIp)
        etPcPort = findViewById(R.id.etPcPort)
        spinnerVideos = findViewById(R.id.spinnerVideos)
        btnUpload = findViewById(R.id.btnUpload)
        tvUploadStatus = findViewById(R.id.tvUploadStatus)
        bottomNav = findViewById(R.id.bottomNav)
        sectionRecording = findViewById(R.id.sectionRecording)
        sectionOfflineRecording = findViewById(R.id.sectionOfflineRecording)
        sectionEarnings = findViewById(R.id.sectionEarnings)
        sectionData = findViewById(R.id.sectionData)
        sectionSettings = findViewById(R.id.sectionSettings)

        findViewById<MaterialButton>(R.id.btnEnableOfflineRecording).setOnClickListener { openOfflineRecording() }
        findViewById<MaterialButton>(R.id.btnOfflineReturn).setOnClickListener { closeOfflineRecording() }
        findViewById<MaterialButton>(R.id.btnOfflineNext).setOnClickListener { onOfflineNext() }

        findViewById<ImageButton>(R.id.btnDismissCameraTip).setOnClickListener {
            findViewById<View>(R.id.layoutCameraTip).visibility = View.GONE
        }
        findViewById<TextView>(R.id.btnDismissQuestionnaire).setOnClickListener {
            findViewById<View>(R.id.cardQuestionnaire).visibility = View.GONE
        }
        findViewById<TextView>(R.id.btnDismissSummary).setOnClickListener {
            findViewById<View>(R.id.cardDailySummary).visibility = View.GONE
        }
        findViewById<TextView>(R.id.btnViewQuestionnaire).setOnClickListener {
            Toast.makeText(this, "Questionnaires (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<TextView>(R.id.btnViewSummary).setOnClickListener {
            Toast.makeText(this, "Daily summary (demo)", Toast.LENGTH_SHORT).show()
        }

        btnPcRecord.setOnClickListener { togglePcRecording() }
        btnPhoneRecord.setOnClickListener { togglePhoneRecording() }
        btnUpload.setOnClickListener { uploadToPc() }
        findViewById<View>(R.id.switchAutoRecording).setOnClickListener {
            Toast.makeText(this, "Auto recording toggle is UI-only for now.", Toast.LENGTH_SHORT).show()
        }

        val recordDir = File(getExternalFilesDir(android.os.Environment.DIRECTORY_MOVIES), "ScreenRecords")
        tvRecordPath.text = getString(R.string.record_path_caption, recordDir.absolutePath)

        findViewById<RadioGroup>(R.id.rgOfflineDevices).check(R.id.rbDeviceDsj)
        findViewById<RadioGroup>(R.id.rgOfflineAcquire).check(R.id.rbPurchase)

        sectionOfflineRecording.post { applyOfflineDeviceThumbnails() }

        setupEarningsHeaderIntro()
        populateEarningsActivityRows()
        setupEarningsAccordion()
        setupDataScreen()
        setupSettingsScreen()

        setupBottomNav()
        refreshVideoList()
        updatePcRecordButtonUi()
        updatePhoneRecordButtonUi()
        spinnerVideos.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                selectedUploadFile = recordingFiles.getOrNull(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedUploadFile = null
            }
        }

        LocalBroadcastManager.getInstance(this).registerReceiver(
            errorReceiver,
            IntentFilter(ScreenRecordService.BROADCAST_ERROR)
        )
    }

    private fun openOfflineRecording() {
        sectionRecording.visibility = View.GONE
        sectionOfflineRecording.visibility = View.VISIBLE
        applyOfflineDeviceThumbnails()
    }

    /** Bitmap drawables as compoundStart used intrinsic size and overflowed; scale top drawable to ~50% of panel width. */
    private fun applyOfflineDeviceThumbnails() {
        sectionOfflineRecording.post {
            val panelW = sectionOfflineRecording.width
            if (panelW <= 0) return@post
            val innerW = panelW - sectionOfflineRecording.paddingStart - sectionOfflineRecording.paddingEnd
            val thumbW = (innerW * 0.5f).toInt().coerceIn(80, 900)
            fun RadioButton.setScaledTopDrawable(@DrawableRes res: Int) {
                val d = AppCompatResources.getDrawable(this@MainActivity, res)?.mutate() ?: return
                val iw = d.intrinsicWidth
                val ih = d.intrinsicHeight
                val ratio = if (iw > 0 && ih > 0) ih.toFloat() / iw else 0.75f
                val thumbH = (thumbW * ratio).toInt().coerceAtLeast(1)
                d.setBounds(0, 0, thumbW, thumbH)
                setCompoundDrawablesRelative(null, d, null, null)
            }
            findViewById<RadioButton>(R.id.rbDeviceDsj).setScaledTopDrawable(R.drawable.device_bodycam)
            findViewById<RadioButton>(R.id.rbDevicePickle).setScaledTopDrawable(R.drawable.device_smartglasses)
            findViewById<RadioButton>(R.id.rbDeviceLooki).setScaledTopDrawable(R.drawable.device_wearable_cam)
        }
    }

    private fun closeOfflineRecording() {
        sectionOfflineRecording.visibility = View.GONE
        sectionRecording.visibility = View.VISIBLE
    }

    private fun onOfflineNext() {
        val dev = findViewById<RadioGroup>(R.id.rgOfflineDevices).checkedRadioButtonId
        val acq = findViewById<RadioGroup>(R.id.rgOfflineAcquire).checkedRadioButtonId
        val devLabel = when (dev) {
            R.id.rbDevicePickle -> "Pickle 1"
            R.id.rbDeviceLooki -> "Looki"
            else -> "DSJ-HLN 19 A1"
        }
        val acqLabel = when (acq) {
            R.id.rbRent -> getString(R.string.rent_freely)
            R.id.rbConnect -> getString(R.string.connect_existing)
            else -> getString(R.string.purchase_online)
        }
        Toast.makeText(this, "Next: $devLabel · $acqLabel (demo)", Toast.LENGTH_SHORT).show()
    }

    private fun setupBottomNav() {
        bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_search -> {
                    Toast.makeText(this, "Search (coming soon)", Toast.LENGTH_SHORT).show()
                    false
                }
                R.id.nav_recording -> {
                    showSection(sectionRecording)
                    true
                }
                R.id.nav_earnings -> {
                    showSection(sectionEarnings)
                    true
                }
                R.id.nav_data -> {
                    showSection(sectionData)
                    true
                }
                R.id.nav_settings -> {
                    showSection(sectionSettings)
                    true
                }
                else -> false
            }
        }
        bottomNav.selectedItemId = R.id.nav_recording
    }

    private fun showSection(target: View) {
        val sections = listOf(sectionRecording, sectionOfflineRecording, sectionEarnings, sectionData, sectionSettings)
        sections.forEach { it.visibility = if (it == target) View.VISIBLE else View.GONE }
    }

    private fun updatePcRecordButtonUi() {
        if (isPcRecording) {
            btnPcRecord.text = getString(R.string.stop_pc_recording)
            btnPcRecord.setIconResource(R.drawable.ic_pause_blue)
            tvPcRecordStatus.text = getString(R.string.pc_record_status_active)
        } else {
            btnPcRecord.text = getString(R.string.start_pc_recording)
            btnPcRecord.setIconResource(R.drawable.ic_play_blue)
            tvPcRecordStatus.text = getString(R.string.pc_record_status_idle)
        }
    }

    private fun updatePhoneRecordButtonUi() {
        if (isPhoneRecording) {
            btnPhoneRecord.text = getString(R.string.stop_record)
            btnPhoneRecord.setIconResource(R.drawable.ic_pause_blue)
        } else {
            btnPhoneRecord.text = getString(R.string.start_record)
            btnPhoneRecord.setIconResource(R.drawable.ic_play_blue)
        }
    }

    private fun togglePcRecording() {
        isPcRecording = !isPcRecording
        updatePcRecordButtonUi()
        if (isPcRecording) {
            Toast.makeText(this, "PC recording: desktop capture will start when API is connected.", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "PC recording: desktop capture will stop when API is connected.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        refreshVideoList()
    }

    override fun onDestroy() {
        super.onDestroy()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(errorReceiver)
    }

    private fun getRecordingsDir(): File {
        return File(getExternalFilesDir(android.os.Environment.DIRECTORY_MOVIES), "ScreenRecords")
    }

    private fun refreshVideoList() {
        val dir = getRecordingsDir()
        recordingFiles = dir.listFiles { _, name -> name.endsWith(".mp4") }
            ?.sortedByDescending { it.lastModified() }
            ?: emptyList()
        val displayItems = if (recordingFiles.isEmpty()) {
            listOf("No recordings yet (record first)")
        } else {
            recordingFiles.map { f ->
                val sizeKb = f.length() / 1024
                "${f.name} (${sizeKb}KB)"
            }
        }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, displayItems)
        spinnerVideos.adapter = adapter
        selectedUploadFile = recordingFiles.firstOrNull()
    }

    private fun togglePhoneRecording() {
        if (isPhoneRecording) {
            stopRecordService()
            isPhoneRecording = false
            updatePhoneRecordButtonUi()
            tvRecordStatus.text = getString(R.string.record_status_idle)
        } else {
            Log.d(TAG, "Requesting screen capture permission...")
            tvRecordStatus.text = "Phone: waiting for permission…"
            val pm = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            val captureIntent = pm.createScreenCaptureIntent()
            projectionLauncher.launch(captureIntent)
        }
    }

    private fun startRecordService(resultCode: Int, data: Intent) {
        Log.d(TAG, "startRecordService: resultCode=$resultCode")
        isPhoneRecording = true
        updatePhoneRecordButtonUi()
        tvRecordStatus.text = "Phone: recording…"
        val intent = Intent(this, ScreenRecordService::class.java).apply {
            action = ScreenRecordService.ACTION_START
            putExtra(ScreenRecordService.EXTRA_RESULT_CODE, resultCode)
            putExtra(ScreenRecordService.EXTRA_RESULT_DATA, data)
        }
        ContextCompat.startForegroundService(this, intent)
    }

    private fun stopRecordService() {
        Log.d(TAG, "stopRecordService")
        val intent = Intent(this, ScreenRecordService::class.java).apply {
            action = ScreenRecordService.ACTION_STOP
        }
        startService(intent)
    }

    private fun uploadToPc() {
        val ip = etPcIp.text?.toString()?.trim() ?: ""
        val portStr = etPcPort.text?.toString()?.trim() ?: "8765"
        val port = portStr.toIntOrNull() ?: 8765
        if (ip.isEmpty()) {
            tvUploadStatus.text = "Enter your PC’s IP address."
            return
        }
        val fileToUpload = selectedUploadFile
        if (fileToUpload == null || !fileToUpload.exists()) {
            tvUploadStatus.text = "Pick a recording from the list above."
            return
        }
        if (fileToUpload.length() == 0L) {
            tvUploadStatus.text = "This file is empty. Record again or choose another file."
            return
        }
        tvUploadStatus.text = "Uploading ${fileToUpload.name} (${fileToUpload.length() / 1024} KB)…"
        btnUpload.isEnabled = false
        CoroutineScope(Dispatchers.Main).launch {
            val result = withContext(Dispatchers.IO) {
                uploadFile("http://$ip:$port/upload", fileToUpload)
            }
            tvUploadStatus.text = result
            btnUpload.isEnabled = true
        }
    }

    private fun uploadFile(url: String, file: File): String {
        return try {
            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(120, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .build()
            val primary = uploadOnce(client, url, file, "file")
            if (primary.first) {
                "Upload OK: ${file.name}"
            } else {
                val fallback = if (primary.second == 400) uploadOnce(client, url, file, "video") else primary
                if (fallback.first) {
                    "Upload OK: ${file.name}"
                } else {
                    val reason = if (fallback.third.isNotBlank()) " — ${fallback.third}" else ""
                    "Upload failed: HTTP ${fallback.second}$reason"
                }
            }
        } catch (e: Exception) {
            "Upload failed: ${e.javaClass.simpleName}: ${e.message}"
        }
    }

    private fun uploadOnce(
        client: OkHttpClient,
        url: String,
        file: File,
        fieldName: String
    ): Triple<Boolean, Int, String> {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                fieldName,
                file.name,
                file.asRequestBody("video/mp4".toMediaType())
            )
            .build()
        val request = Request.Builder().url(url).post(body).build()
        client.newCall(request).execute().use { response ->
            val code = response.code
            val detail = response.body?.string().orEmpty()
            return Triple(response.isSuccessful, code, detail)
        }
    }

    private fun setupEarningsHeaderIntro() {
        findViewById<TextView>(R.id.tv_earnings_survey_intro).text =
            HtmlCompat.fromHtml(getString(R.string.earnings_survey_intro_html), HtmlCompat.FROM_HTML_MODE_LEGACY)
    }

    private fun populateEarningsActivityRows() {
        val rec = findViewById<LinearLayout>(R.id.content_earnings_recommendation)
        val diary = findViewById<LinearLayout>(R.id.content_earnings_diary)
        rec.removeAllViews()
        diary.removeAllViews()
        val inflater = layoutInflater
        val recRows = listOf(
            "19 March, 2026 Thur" to "9h 35 min",
            "18 March, 2026 Wed" to "8h 20 min",
            "17 March, 2026 Tue" to "7h 55 min",
            "16 March, 2026 Mon" to "9h 02 min"
        )
        for ((dayTitle, dur) in recRows) {
            val row = inflater.inflate(R.layout.item_earnings_activity_row, rec, false)
            row.findViewById<TextView>(R.id.tv_earnings_row_title).text = dayTitle
            row.findViewById<TextView>(R.id.tv_earnings_row_subtitle).text =
                getString(R.string.earnings_log_online, dur)
            row.setOnClickListener {
                Toast.makeText(this, dayTitle, Toast.LENGTH_SHORT).show()
            }
            rec.addView(row)
        }
        val diaryRows = listOf(
            "15 March, 2026 Sun" to "Daily reflection · 3 highlights",
            "14 March, 2026 Sat" to "Voice note · evening summary",
            "13 March, 2026 Fri" to "Auto summary · workday",
            "12 March, 2026 Thu" to "Mood tracker · check-in"
        )
        for ((dayTitle, sub) in diaryRows) {
            val row = inflater.inflate(R.layout.item_earnings_activity_row, diary, false)
            row.findViewById<TextView>(R.id.tv_earnings_row_title).text = dayTitle
            row.findViewById<TextView>(R.id.tv_earnings_row_subtitle).text = sub
            row.setOnClickListener {
                Toast.makeText(this, dayTitle, Toast.LENGTH_SHORT).show()
            }
            diary.addView(row)
        }
    }

    private enum class DataDimension { TIME, TOPIC, EVENT, VALUE }

    private data class DataSessionRow(val title: String, val subtitle: String)

    private data class DataSection(val heading: String, val rows: List<DataSessionRow>)

    private fun setupDataScreen() {
        findViewById<RadioGroup>(R.id.rgDataDimension).setOnCheckedChangeListener { _, checkedId ->
            val dim = when (checkedId) {
                R.id.rbDataTopic -> DataDimension.TOPIC
                R.id.rbDataEvent -> DataDimension.EVENT
                R.id.rbDataValue -> DataDimension.VALUE
                else -> DataDimension.TIME
            }
            populateDataList(dim)
        }
        populateDataList(DataDimension.TIME)
    }

    private fun populateDataList(dim: DataDimension) {
        val container = findViewById<LinearLayout>(R.id.llDataList)
        container.removeAllViews()
        val inflater = layoutInflater
        for (section in sectionsForDimension(dim)) {
            val header = inflater.inflate(R.layout.item_data_header, container, false)
            header.findViewById<TextView>(R.id.tv_data_header).text = section.heading
            container.addView(header)
            for (row in section.rows) {
                val item = inflater.inflate(R.layout.item_data_session_row, container, false)
                item.findViewById<TextView>(R.id.tv_data_row_title).text = row.title
                item.findViewById<TextView>(R.id.tv_data_row_subtitle).text = row.subtitle
                item.setOnClickListener {
                    Toast.makeText(this, row.title, Toast.LENGTH_SHORT).show()
                }
                container.addView(item)
            }
        }
    }

    private fun onlineLine(hours: String, minutes: String): String =
        getString(R.string.earnings_log_online, "$hours $minutes")

    private fun sectionsForDimension(dim: DataDimension): List<DataSection> {
        return when (dim) {
            DataDimension.TIME -> listOf(
                DataSection(
                    getString(R.string.data_section_this_week),
                    listOf(
                        DataSessionRow("19 March, 2026 Thur", onlineLine("9h 35", "min")),
                        DataSessionRow("18 March, 2026 Wed", onlineLine("8h 20", "min")),
                        DataSessionRow("17 March, 2026 Tue", onlineLine("7h 55", "min")),
                        DataSessionRow("16 March, 2026 Mon", onlineLine("9h 02", "min"))
                    )
                ),
                DataSection(
                    getString(R.string.data_section_this_month),
                    listOf(
                        DataSessionRow("15 March, 2026 Sun", onlineLine("6h 40", "min")),
                        DataSessionRow("14 March, 2026 Sat", onlineLine("5h 15", "min"))
                    )
                )
            )
            DataDimension.TOPIC -> listOf(
                DataSection(
                    getString(R.string.data_section_work),
                    listOf(
                        DataSessionRow("Deep work block · Mon", onlineLine("4h 10", "min")),
                        DataSessionRow("Email & admin · Tue", onlineLine("2h 05", "min"))
                    )
                ),
                DataSection(
                    getString(R.string.data_section_social),
                    listOf(
                        DataSessionRow("Calls & messages · Wed", onlineLine("1h 30", "min"))
                    )
                ),
                DataSection(
                    getString(R.string.data_section_learning),
                    listOf(
                        DataSessionRow("Course playback · Thu", onlineLine("3h 00", "min"))
                    )
                )
            )
            DataDimension.EVENT -> listOf(
                DataSection(
                    getString(R.string.data_section_meetings),
                    listOf(
                        DataSessionRow("Stand-up · 10:00", onlineLine("0h 45", "min")),
                        DataSessionRow("Project review · 15:00", onlineLine("1h 20", "min"))
                    )
                ),
                DataSection(
                    getString(R.string.data_section_travel),
                    listOf(
                        DataSessionRow("Commute · morning", onlineLine("0h 35", "min")),
                        DataSessionRow("Commute · evening", onlineLine("0h 42", "min"))
                    )
                )
            )
            DataDimension.VALUE -> listOf(
                DataSection(
                    getString(R.string.data_section_productivity),
                    listOf(
                        DataSessionRow("Focus score · week", "High · Online Recording"),
                        DataSessionRow("Tasks completed", "12 items · Online Recording")
                    )
                ),
                DataSection(
                    getString(R.string.data_section_wellness),
                    listOf(
                        DataSessionRow("Breaks & movement", "On track · Online Recording"),
                        DataSessionRow("Evening wind-down", onlineLine("1h 10", "min"))
                    )
                )
            )
        }
    }

    private fun setupSettingsScreen() {
        val agentDataDir = File(getExternalFilesDir(null), "AgentData")
        findViewById<TextView>(R.id.tv_settings_storage_data).text =
            "${agentDataDir.absolutePath} · 6.7 GB"

        findViewById<ImageButton>(R.id.btn_settings_edit_phone).setOnClickListener {
            Toast.makeText(this, "Edit phone label (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<ImageButton>(R.id.btn_settings_edit_pc).setOnClickListener {
            Toast.makeText(this, "Edit PC label (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<View>(R.id.row_settings_offline).setOnClickListener {
            bottomNav.selectedItemId = R.id.nav_recording
            showSection(sectionRecording)
            openOfflineRecording()
        }
        findViewById<ImageButton>(R.id.btn_settings_edit_gui).setOnClickListener {
            Toast.makeText(this, "GUI Agent: ${getString(R.string.settings_gui_agent_value)} (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<ImageButton>(R.id.btn_settings_edit_vlm).setOnClickListener {
            Toast.makeText(this, "Multi-modal LLM (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<ImageButton>(R.id.btn_settings_edit_data).setOnClickListener {
            Toast.makeText(this, "Storage: ${agentDataDir.absolutePath} (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<View>(R.id.row_settings_app_survey).setOnClickListener {
            startActivity(Intent(this, AutoFilledSurveyActivity::class.java))
        }
        findViewById<View>(R.id.row_settings_app_rec).setOnClickListener {
            Toast.makeText(this, "Personal recommendation (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<View>(R.id.row_settings_app_diary).setOnClickListener {
            Toast.makeText(this, "AI-powered diary (demo)", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupEarningsAccordion() {
        val contentSurvey = findViewById<View>(R.id.content_earnings_survey)
        val chevS = findViewById<TextView>(R.id.chevron_earnings_survey)
        val contentRec = findViewById<View>(R.id.content_earnings_recommendation)
        val chevR = findViewById<TextView>(R.id.chevron_earnings_recommendation)
        val contentDiary = findViewById<View>(R.id.content_earnings_diary)
        val chevD = findViewById<TextView>(R.id.chevron_earnings_diary)

        findViewById<View>(R.id.header_earnings_survey).setOnClickListener {
            earningsSurveyExpanded = !earningsSurveyExpanded
            contentSurvey.visibility = if (earningsSurveyExpanded) View.VISIBLE else View.GONE
            chevS.text = if (earningsSurveyExpanded) "▲" else "▼"
        }
        findViewById<View>(R.id.header_earnings_recommendation).setOnClickListener {
            earningsRecExpanded = !earningsRecExpanded
            contentRec.visibility = if (earningsRecExpanded) View.VISIBLE else View.GONE
            chevR.text = if (earningsRecExpanded) "▲" else "▼"
        }
        findViewById<View>(R.id.header_earnings_diary).setOnClickListener {
            earningsDiaryExpanded = !earningsDiaryExpanded
            contentDiary.visibility = if (earningsDiaryExpanded) View.VISIBLE else View.GONE
            chevD.text = if (earningsDiaryExpanded) "▲" else "▼"
        }

        findViewById<MaterialButton>(R.id.btn_review_surveys).setOnClickListener {
            startActivity(Intent(this, AutoFilledSurveyActivity::class.java))
        }
        findViewById<View>(R.id.row_earnings_more_apps).setOnClickListener {
            Toast.makeText(this, "More applications (demo)", Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
