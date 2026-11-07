plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.meevo.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.meevo.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += "lib/*/libonnxruntime.so"
        }
        resources.excludes += "META-INF/*.version"
        resources.excludes += "META-INF/LICENSE*"
        resources.excludes += "META-INF/NOTICE*"
    }

    androidResources {
        noCompress += listOf("onnx", "data", "wav", "jinja")
    }

    lint { abortOnError = false }
}

dependencies {
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.aar"))))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.0")
}
