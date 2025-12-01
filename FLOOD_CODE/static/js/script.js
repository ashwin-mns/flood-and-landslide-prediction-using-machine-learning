// Global variables
let predictionData = null;

$(document).ready(function() {
    // Initialize tooltips
    $('[data-bs-toggle="tooltip"]').tooltip();

    // Handle prediction form submission
    $('#predictionForm').on('submit', function(e) {
        e.preventDefault();
        makePrediction();
    });

    // Handle delete buttons in dashboard
    $(document).on('click', '.delete-btn', function() {
        const predictionId = $(this).data('id');
        deletePrediction(predictionId);
    });

    // Load result if we have prediction data
    if (window.location.pathname === '/result' && predictionData) {
        displayResult(predictionData);
    }
});

// Make prediction
function makePrediction() {
    const formData = {
        water_level: $('#water_level').val(),
        rain: $('#rain').val(),
        soil_moisture: $('#soil_moisture').val()
    };

    // Validate inputs
    if (!validateInputs(formData)) {
        return;
    }

    // Show loading modal
    $('#loadingModal').modal('show');

    // Make AJAX request
    $.ajax({
        url: '/predict',
        type: 'POST',
        data: formData,
        success: function(response) {
            $('#loadingModal').modal('hide');
            
            if (response.success) {
                predictionData = response;
                window.location.href = '/result';
            } else {
                showAlert('Error: ' + response.error, 'danger');
            }
        },
        error: function(xhr) {
            $('#loadingModal').modal('hide');
            showAlert('An error occurred while making the prediction.', 'danger');
        }
    });
}

// Validate form inputs
function validateInputs(data) {
    const waterLevel = parseFloat(data.water_level);
    const rain = parseFloat(data.rain);
    const soilMoisture = parseFloat(data.soil_moisture);

    if (waterLevel < 0 || waterLevel > 20) {
        showAlert('Water level must be between 0 and 20.', 'warning');
        return false;
    }

    if (rain < 0 || rain > 4095) {
        showAlert('Rain sensor value must be between 0 and 4095.', 'warning');
        return false;
    }

    if (soilMoisture < 0 || soilMoisture > 4095) {
        showAlert('Soil moisture must be between 0 and 4095.', 'warning');
        return false;
    }

    return true;
}

// Display result on result page
function displayResult(data) {
    const resultCard = $('#resultCard');
    const prediction = data.prediction;
    
    let iconClass, titleClass, titleText, message;

    switch(prediction) {
        case 'no flood':
            iconClass = 'fas fa-check-circle result-success';
            titleClass = 'text-success';
            titleText = 'NO FLOOD DETECTED';
            message = 'Conditions are safe. No immediate flood risk detected.';
            break;
        case 'suspect':
            iconClass = 'fas fa-exclamation-triangle result-warning';
            titleClass = 'text-warning';
            titleText = 'SUSPECT CONDITIONS';
            message = 'Potential risk detected. Monitor the situation closely.';
            break;
        case 'flood':
            iconClass = 'fas fa-exclamation-circle result-danger';
            titleClass = 'text-danger';
            titleText = 'FLOOD DETECTED';
            message = 'High flood risk detected. Take necessary precautions immediately.';
            break;
        default:
            iconClass = 'fas fa-question-circle text-secondary';
            titleClass = 'text-secondary';
            titleText = 'UNKNOWN';
            message = 'Unable to determine flood condition.';
    }

    const resultHtml = `
        <div class="card-header ${titleClass} bg-transparent text-center py-4 border-0">
            <i class="${iconClass} result-icon"></i>
            <h2 class="display-6 fw-bold">${titleText}</h2>
        </div>
        <div class="card-body text-center p-4">
            <p class="lead mb-4">${message}</p>
            
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Water Level</h6>
                            <h4 class="text-primary">${data.water_level}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Rain</h6>
                            <h4 class="text-info">${data.rain}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-title">Soil Moisture</h6>
                            <h4 class="text-warning">${data.soil_moisture}</h4>
                        </div>
                    </div>
                </div>
            </div>

            ${data.probability ? `
            <div class="mb-4">
                <h5>Prediction Confidence</h5>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar ${titleClass.replace('text-', 'bg-')}" 
                         style="width: ${data.probability * 100}%">
                        ${(data.probability * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
            ` : ''}

            <div class="alert ${titleClass.replace('text-', 'alert-')}">
                <i class="fas fa-info-circle me-2"></i>
                <strong>Recommendation:</strong> ${getRecommendation(prediction)}
            </div>
        </div>
    `;

    resultCard.html(resultHtml);
    resultCard.addClass('animate__animated animate__zoomIn');
}

// Get recommendation based on prediction
function getRecommendation(prediction) {
    switch(prediction) {
        case 'no flood':
            return 'Continue normal monitoring. Conditions are stable.';
        case 'suspect':
            return 'Increase monitoring frequency and prepare contingency plans.';
        case 'flood':
            return 'Activate emergency protocols and consider evacuation if necessary.';
        default:
            return 'Review sensor data and system status.';
    }
}

// Delete prediction
function deletePrediction(predictionId) {
    if (!confirm('Are you sure you want to delete this prediction?')) {
        return;
    }

    $.ajax({
        url: `/delete_prediction/${predictionId}`,
        type: 'GET',
        success: function(response) {
            if (response.success) {
                $(`button[data-id="${predictionId}"]`).closest('tr').fadeOut(300, function() {
                    $(this).remove();
                    showAlert('Prediction deleted successfully.', 'success');
                });
            } else {
                showAlert('Failed to delete prediction.', 'danger');
            }
        },
        error: function() {
            showAlert('An error occurred while deleting the prediction.', 'danger');
        }
    });
}

// Show alert message
function showAlert(message, type) {
    const alertClass = `alert-${type}`;
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show animate__animated animate__fadeIn">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    $('.container.mt-3').prepend(alertHtml);
    
    // Auto remove alert after 5 seconds
    setTimeout(() => {
        $('.alert').alert('close');
    }, 5000);
}

// Input validation with visual feedback
$('input[type="number"]').on('input', function() {
    const value = parseFloat($(this).val());
    const min = parseFloat($(this).attr('min'));
    const max = parseFloat($(this).attr('max'));
    
    if (value < min || value > max) {
        $(this).addClass('is-invalid');
    } else {
        $(this).removeClass('is-invalid').addClass('is-valid');
    }
});

// Add animation to elements when they come into view
function animateOnScroll() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    elements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const elementVisible = 150;
        
        if (elementTop < window.innerHeight - elementVisible) {
            element.classList.add('animate__animated', 'animate__fadeInUp');
        }
    });
}

// Initialize scroll animations
window.addEventListener('scroll', animateOnScroll);
animateOnScroll(); // Initial check