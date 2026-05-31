package com.screenrecorder.mvp

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class SurveyListAdapter(
    private val items: List<SurveyListItem>,
    private val onItemClick: (SurveyListItem) -> Unit
) : RecyclerView.Adapter<SurveyListAdapter.VH>() {

    class VH(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val title: TextView = itemView.findViewById(R.id.tv_survey_row_title)
        val price: TextView = itemView.findViewById(R.id.tv_survey_row_price)
        val questions: TextView = itemView.findViewById(R.id.tv_survey_row_questions)
        val time: TextView = itemView.findViewById(R.id.tv_survey_row_time)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_survey_list_row, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = items[position]
        holder.title.text = item.title
        holder.price.text = item.price
        holder.questions.text = item.questionsLabel
        holder.time.text = item.timeAgo
        holder.itemView.setOnClickListener { onItemClick(item) }
    }

    override fun getItemCount(): Int = items.size
}
