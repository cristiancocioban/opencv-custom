ANDROID_NATIVE_API_LEVEL = int(os.environ.get('ANDROID_NATIVE_API_LEVEL', 24))
cmake_common_vars = {
    'ANDROID_COMPILE_SDK_VERSION': os.environ.get('ANDROID_COMPILE_SDK_VERSION', 34),
    'ANDROID_TARGET_SDK_VERSION': os.environ.get('ANDROID_TARGET_SDK_VERSION', 34),
    'ANDROID_MIN_SDK_VERSION': os.environ.get('ANDROID_MIN_SDK_VERSION', ANDROID_NATIVE_API_LEVEL),
    'ANDROID_GRADLE_PLUGIN_VERSION': '7.3.1',
    'GRADLE_VERSION': '7.5.1',
    'KOTLIN_PLUGIN_VERSION': '1.8.20',
}
ABIs = [
    ABI("3", "arm64-v8a", None, ndk_api_level=ANDROID_NATIVE_API_LEVEL, cmake_vars=cmake_common_vars),
    ABI("5", "x86_64",    None, ndk_api_level=ANDROID_NATIVE_API_LEVEL, cmake_vars=cmake_common_vars),
]
