import java.io.File

interface AnalysisPipeline {
    fun run(runFile: File): Error
    fun getName(): String
}