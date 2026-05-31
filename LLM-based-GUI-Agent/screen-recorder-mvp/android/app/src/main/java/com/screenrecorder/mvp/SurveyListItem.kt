package com.screenrecorder.mvp

data class SurveyListItem(
    val title: String,
    val price: String,
    val questionsLabel: String,
    val timeAgo: String
)

object SurveyDemo {
    val filledSurveys: List<SurveyListItem> = List(5) {
        SurveyListItem(
            title = "Coffee Market Survey",
            price = "$23",
            questionsLabel = "15 Questions",
            timeAgo = "6 hours ago"
        )
    }
}
