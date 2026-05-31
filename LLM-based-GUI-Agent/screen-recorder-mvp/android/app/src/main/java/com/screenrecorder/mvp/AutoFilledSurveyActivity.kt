package com.screenrecorder.mvp

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar

class AutoFilledSurveyActivity : AppCompatActivity() {

    private lateinit var recycler: RecyclerView
    private lateinit var empty: TextView
    private lateinit var tabFilled: TextView
    private lateinit var tabSubmitted: TextView
    private lateinit var tabOpen: TextView

    private enum class SurveyTab { Filled, Submitted, Open }

    private var currentTab = SurveyTab.Filled

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_auto_filled_survey)

        findViewById<MaterialToolbar>(R.id.toolbar_survey_list).setNavigationOnClickListener {
            onBackPressedDispatcher.onBackPressed()
        }

        recycler = findViewById(R.id.recycler_survey_list)
        empty = findViewById(R.id.tv_survey_empty)
        tabFilled = findViewById(R.id.tab_survey_filled)
        tabSubmitted = findViewById(R.id.tab_survey_submitted)
        tabOpen = findViewById(R.id.tab_survey_open)

        recycler.layoutManager = LinearLayoutManager(this)
        val adapter = SurveyListAdapter(SurveyDemo.filledSurveys) { item ->
            startActivity(
                Intent(this, SurveyDetailActivity::class.java)
                    .putExtra(SurveyDetailActivity.EXTRA_SURVEY_TITLE, item.title)
            )
        }
        recycler.adapter = adapter

        tabFilled.setOnClickListener { selectTab(SurveyTab.Filled) }
        tabSubmitted.setOnClickListener { selectTab(SurveyTab.Submitted) }
        tabOpen.setOnClickListener { selectTab(SurveyTab.Open) }

        selectTab(SurveyTab.Filled)
    }

    private fun selectTab(tab: SurveyTab) {
        currentTab = tab
        val selectedBg = ContextCompat.getDrawable(this, R.drawable.bg_survey_tab_selected)
        tabFilled.background = if (tab == SurveyTab.Filled) selectedBg else null
        tabSubmitted.background = if (tab == SurveyTab.Submitted) selectedBg else null
        tabOpen.background = if (tab == SurveyTab.Open) selectedBg else null

        val blue = ContextCompat.getColor(this, R.color.mock_ios_blue)
        val black = ContextCompat.getColor(this, R.color.mock_text)
        tabFilled.setTextColor(if (tab == SurveyTab.Filled) blue else black)
        tabSubmitted.setTextColor(if (tab == SurveyTab.Submitted) blue else black)
        tabOpen.setTextColor(if (tab == SurveyTab.Open) blue else black)

        when (tab) {
            SurveyTab.Filled -> {
                recycler.visibility = View.VISIBLE
                empty.visibility = View.GONE
            }
            SurveyTab.Submitted, SurveyTab.Open -> {
                recycler.visibility = View.GONE
                empty.visibility = View.VISIBLE
            }
        }
    }
}
