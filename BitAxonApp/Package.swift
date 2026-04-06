// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "BitAxonApp",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.29.1"..<"0.30.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", from: "2.21.2"),
    ],
    targets: [
        .executableTarget(
            name: "BitAxonApp",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
            ],
            path: "BitAxonApp"
        ),
        .testTarget(
            name: "EquivalenceTests",
            dependencies: ["BitAxonApp"],
            path: "Tests/EquivalenceTests",
            exclude: ["EquivalenceTestSupport/export_reference.py"],
            resources: [
                .copy("EquivalenceTestSupport/reference"),
            ]
        ),
    ]
)
