import java.io.File

data class Error(
        val message: String = "",
        val value: File? = null
)
