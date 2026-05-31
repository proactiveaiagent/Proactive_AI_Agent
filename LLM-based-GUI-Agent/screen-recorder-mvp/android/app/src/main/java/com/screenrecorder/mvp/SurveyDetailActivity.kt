package com.screenrecorder.mvp

import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton

class SurveyDetailActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_survey_detail)

        val title = intent.getStringExtra(EXTRA_SURVEY_TITLE).orEmpty()
        val toolbar = findViewById<MaterialToolbar>(R.id.toolbar_survey_detail)
        toolbar.title = if (title.isNotBlank()) title else getString(R.string.earnings_acc_survey)
        toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }

        val blue = ContextCompat.getColor(this, R.color.mock_ios_blue)
        val black = ContextCompat.getColor(this, R.color.mock_text)
        val selectedBg = ContextCompat.getDrawable(this, R.drawable.bg_survey_tab_selected)

        val tabFilled = findViewById<TextView>(R.id.tab_detail_filled)
        val tabSubmitted = findViewById<TextView>(R.id.tab_detail_submitted)
        val tabOpen = findViewById<TextView>(R.id.tab_detail_open)

        fun styleTab(selected: TextView) {
            tabFilled.background = if (selected == tabFilled) selectedBg else null
            tabSubmitted.background = if (selected == tabSubmitted) selectedBg else null
            tabOpen.background = if (selected == tabOpen) selectedBg else null
            tabFilled.setTextColor(if (selected == tabFilled) blue else black)
            tabSubmitted.setTextColor(if (selected == tabSubmitted) blue else black)
            tabOpen.setTextColor(if (selected == tabOpen) blue else black)
        }
        styleTab(tabFilled)
        tabFilled.setOnClickListener { styleTab(tabFilled) }
        tabSubmitted.setOnClickListener { styleTab(tabSubmitted) }
        tabOpen.setOnClickListener { styleTab(tabOpen) }

        findViewById<MaterialButton>(R.id.btn_survey_discard).setOnClickListener {
            Toast.makeText(this, "Discard (demo)", Toast.LENGTH_SHORT).show()
        }
        findViewById<MaterialButton>(R.id.btn_survey_submit).setOnClickListener {
            Toast.makeText(this, "Submit (demo)", Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        const val EXTRA_SURVEY_TITLE = "survey_title"
    }
}
