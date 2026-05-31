use axum::Router;
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum_extra::extract::Multipart;
use clap::Parser;
use image::ImageReader;
use std::io::Cursor;
use tract_onnx::prelude::*;

/// Digit classifier, trained on the mnist dataset
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> TractResult<()> {
    // Load the ONNX model
    let model = tract_onnx::onnx()
        .model_for_path("mnist.onnx")?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, 28, 28)),
        )?
        .into_optimized()?
        .into_runnable()?;

    let app = Router::new()
        .route("/", get(root))
        .route("/image", post(upload_image))
        .with_state(model);

    let port = Args::parse().port;

    println!("Listening on port {port}");
    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();

    axum::serve(listener, app).await.unwrap();
    Ok(())
}

async fn root() -> Html<&'static str> {
    Html(
        r#"
        <!DOCTYPE html>
        <html>
        <body>
            <h1>Upload PNG Image</h1>
            <form action="/image" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/png"/>
                <input type="submit" value="Upload"/>
            </form>
        </body>
        </html>
        "#,
    )
}

async fn upload_image(
    State(model): State<
        SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    >,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Only look for the first file field
    while let Some(field) = multipart.next_field().await.unwrap() {
        if field.name() == Some("file") {
            println!(
                "Hit on /image with name {}",
                field.file_name().unwrap_or("<empty>")
            );
            let data = field.bytes().await.unwrap();
            let image = match load_image_as_f32_vector(&data) {
                Ok(image) => image,
                Err(e) => {
                    return Html(format!(
                        "<h2>Error processing image!</h2><p>{}</p><a href='/'>Back</a>",
                        e
                    ));
                }
            };
            // Run inference
            let input = tract_ndarray::Array4::from_shape_vec((1, 1, 28, 28), image)
                .unwrap()
                .into_tensor();
            let result = model.run(tvec!(input.into())).unwrap();

            // Get predicted class
            let tensor = result[0].to_array_view::<f32>().unwrap();
            let best = tensor
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if !data.is_empty() {
                let random_number = best.0;
                return Html(format!(
                    "<h2>Upload successful!</h2>
                    <p>Predicted number: [{}]</p>
                    <a href='/'>Back</a>",
                    random_number
                ));
            }
        }
    }
    Html("<h2>Upload failed!</h2><a href='/'>Back</a>".to_string())
}

pub fn load_image_as_f32_vector(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("Failed to read image format: {}", e))?
        .decode()
        .map_err(|e| format!("Failed to decode image: {}", e))?;

    // Convert to grayscale (Luma)
    let gray = img.to_luma8();

    // Ensure the image is 28x28 (resize if necessary)
    let resized = if gray.dimensions() != (28, 28) {
        image::imageops::resize(&gray, 28, 28, image::imageops::FilterType::Nearest)
    } else {
        gray
    };

    // Convert to f32 and normalize
    let data: Vec<f32> = resized.pixels().map(|p| p[0] as f32 / 255.0).collect();

    Ok(data)
}
