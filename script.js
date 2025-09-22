// Healthcare Chatbot JavaScript
class HealthBot {
    constructor() {
        this.currentTab = 'chat';
        this.appointments = JSON.parse(localStorage.getItem('appointments')) || [];
        this.medications = JSON.parse(localStorage.getItem('medications')) || [];
        this.vitals = JSON.parse(localStorage.getItem('vitals')) || [];
        this.chatHistory = [];
        
        this.initializeEventListeners();
        this.showDisclaimerModal();
        this.updateTimestamps();
        this.loadStoredData();
    }

    initializeEventListeners() {
        // Disclaimer modal
        document.getElementById('acceptDisclaimer').addEventListener('click', () => {
            this.hideDisclaimerModal();
        });

        // Navigation tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.closest('.nav-tab').dataset.tab);
            });
        });

        // Chat functionality
        document.getElementById('sendBtn').addEventListener('click', () => {
            this.sendMessage();
        });

        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Quick action buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const message = e.target.dataset.message;
                document.getElementById('chatInput').value = message;
                this.sendMessage();
            });
        });

        // Symptom checker
        document.getElementById('severityLevel').addEventListener('input', (e) => {
            document.getElementById('severityValue').textContent = e.target.value;
        });

        document.getElementById('checkSymptomsBtn').addEventListener('click', () => {
            this.checkSymptoms();
        });

        // Appointments
        document.getElementById('scheduleBtn').addEventListener('click', () => {
            this.scheduleAppointment();
        });

        // Set minimum date to today
        const today = new Date().toISOString().split('T')[0];
        document.getElementById('preferredDate').min = today;

        // Health records
        document.getElementById('saveVitalsBtn').addEventListener('click', () => {
            this.saveVitals();
        });

        document.getElementById('addMedicationBtn').addEventListener('click', () => {
            this.addMedication();
        });

        // Health info categories
        document.querySelectorAll('.info-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const category = e.target.closest('.info-card').dataset.category;
                this.showHealthInfo(category);
            });
        });

        // Emergency modal
        document.getElementById('emergencyBtn').addEventListener('click', () => {
            this.showEmergencyModal();
        });

        document.getElementById('closeEmergency').addEventListener('click', () => {
            this.hideEmergencyModal();
        });

        // Close modals when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
    }

    showDisclaimerModal() {
        document.getElementById('disclaimerModal').style.display = 'block';
    }

    hideDisclaimerModal() {
        document.getElementById('disclaimerModal').style.display = 'none';
    }

    switchTab(tabName) {
        // Update active tab
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update active content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');

        this.currentTab = tabName;
    }

    sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');
        input.value = '';

        // Process message and generate response
        setTimeout(() => {
            const response = this.generateResponse(message);
            this.addMessage(response, 'bot');
        }, 500);
    }

    addMessage(content, sender) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = `
            <p>${content}</p>
            <small class="timestamp">${this.getCurrentTime()}</small>
        `;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store in chat history
        this.chatHistory.push({ content, sender, timestamp: new Date() });
    }

    generateResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        // Emergency keywords
        const emergencyKeywords = ['emergency', 'urgent', 'chest pain', 'can\'t breathe', 'bleeding', 'unconscious', 'help', '911'];
        if (emergencyKeywords.some(keyword => lowerMessage.includes(keyword))) {
            return `🚨 <strong>This sounds like a medical emergency!</strong><br><br>
                   If you're experiencing a medical emergency, please:<br>
                   • Call 911 immediately<br>
                   • Go to the nearest emergency room<br>
                   • Call poison control: 1-800-222-1222<br><br>
                   I'm here to provide information, but emergency services should be your first priority.`;
        }

        // Symptom-related responses
        if (lowerMessage.includes('symptom') || lowerMessage.includes('pain') || lowerMessage.includes('hurt')) {
            return `I understand you're experiencing symptoms. I'd recommend using our Symptom Checker tab for a more detailed assessment.<br><br>
                   Some general advice:<br>
                   • Monitor your symptoms<br>
                   • Stay hydrated<br>
                   • Rest when possible<br>
                   • Contact your healthcare provider if symptoms worsen<br><br>
                   <em>Remember: This is not a diagnosis. Please consult a healthcare professional for proper medical advice.</em>`;
        }

        // Appointment scheduling
        if (lowerMessage.includes('appointment') || lowerMessage.includes('schedule') || lowerMessage.includes('doctor')) {
            return `I can help you schedule an appointment! Please use the Appointments tab to:<br>
                   • Select appointment type<br>
                   • Choose your preferred date and time<br>
                   • Provide reason for visit<br><br>
                   Available appointment types:<br>
                   • General consultation<br>
                   • Follow-up appointments<br>
                   • Regular check-ups<br>
                   • Specialist consultations<br>
                   • Urgent care`;
        }

        // Health tips
        if (lowerMessage.includes('health tip') || lowerMessage.includes('advice') || lowerMessage.includes('wellness')) {
            const tips = [
                "💧 Stay hydrated by drinking at least 8 glasses of water daily",
                "🏃‍♂️ Aim for 30 minutes of moderate exercise most days of the week",
                "😴 Get 7-9 hours of quality sleep each night",
                "🥗 Eat a balanced diet rich in fruits, vegetables, and whole grains",
                "🧘‍♀️ Practice stress management through meditation or deep breathing",
                "🚭 Avoid smoking and limit alcohol consumption",
                "☀️ Get some sunlight for vitamin D, but use sunscreen",
                "👥 Maintain social connections for mental health"
            ];
            const randomTip = tips[Math.floor(Math.random() * tips.length)];
            return `Here's a health tip for you:<br><br>${randomTip}<br><br>
                   For more comprehensive health information, check out our Health Info tab!`;
        }

        // Medication reminders
        if (lowerMessage.includes('medication') || lowerMessage.includes('medicine') || lowerMessage.includes('pill')) {
            return `I can help you manage your medications! Use the Health Records tab to:<br>
                   • Add medications with dosage information<br>
                   • Set frequency reminders<br>
                   • Track your medication history<br><br>
                   <strong>Important reminders:</strong><br>
                   • Take medications as prescribed<br>
                   • Don't stop medications without consulting your doctor<br>
                   • Keep an updated list for healthcare visits<br>
                   • Store medications safely and check expiration dates`;
        }

        // Mental health
        if (lowerMessage.includes('stress') || lowerMessage.includes('anxiety') || lowerMessage.includes('depression') || lowerMessage.includes('mental')) {
            return `Mental health is just as important as physical health. Here are some resources:<br><br>
                   <strong>Immediate help:</strong><br>
                   • Crisis Text Line: Text HOME to 741741<br>
                   • National Suicide Prevention Lifeline: 988<br><br>
                   <strong>Self-care tips:</strong><br>
                   • Practice deep breathing exercises<br>
                   • Maintain a regular sleep schedule<br>
                   • Stay connected with friends and family<br>
                   • Consider speaking with a mental health professional<br><br>
                   <em>If you're having thoughts of self-harm, please reach out for help immediately.</em>`;
        }

        // Default response
        const defaultResponses = [
            "I'm here to help with your health-related questions! You can ask me about symptoms, schedule appointments, get health tips, or use our various tools.",
            "Feel free to explore the different tabs - I can help with symptom checking, appointment scheduling, health records, and general health information.",
            "I'm designed to provide general health information and assistance. For specific medical advice, please consult with a healthcare professional.",
            "How can I assist you with your health today? I can provide information on a wide range of health topics and help you use our tools."
        ];
        
        return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
    }

    checkSymptoms() {
        const symptoms = document.getElementById('symptomInput').value.trim();
        const duration = document.getElementById('symptomDuration').value;
        const severity = document.getElementById('severityLevel').value;

        if (!symptoms || !duration) {
            alert('Please fill in all required fields.');
            return;
        }

        const resultsDiv = document.getElementById('symptomResults');
        const analysisDiv = document.getElementById('symptomAnalysis');

        // Simple symptom analysis (in real app, this would use proper medical APIs)
        let analysis = this.analyzeSymptoms(symptoms, duration, severity);
        
        analysisDiv.innerHTML = analysis;
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    analyzeSymptoms(symptoms, duration, severity) {
        const lowerSymptoms = symptoms.toLowerCase();
        let urgencyLevel = 'low';
        let recommendations = [];
        let warnings = [];

        // Check for high-severity symptoms
        const emergencySymptoms = ['chest pain', 'difficulty breathing', 'severe bleeding', 'unconscious', 'severe headache'];
        const urgentSymptoms = ['fever', 'persistent vomiting', 'severe pain', 'difficulty swallowing'];
        
        if (emergencySymptoms.some(symptom => lowerSymptoms.includes(symptom)) || severity >= 8) {
            urgencyLevel = 'emergency';
            warnings.push('🚨 These symptoms may require immediate medical attention. Consider calling 911 or going to an emergency room.');
        } else if (urgentSymptoms.some(symptom => lowerSymptoms.includes(symptom)) || severity >= 6) {
            urgencyLevel = 'urgent';
            warnings.push('⚠️ These symptoms should be evaluated by a healthcare professional soon.');
        }

        // Duration-based recommendations
        if (duration === 'weeks') {
            recommendations.push('Symptoms lasting this long should be evaluated by a healthcare professional.');
        }

        // General recommendations
        recommendations.push('Monitor your symptoms and note any changes.');
        recommendations.push('Stay hydrated and get adequate rest.');
        recommendations.push('Keep a symptom diary to track patterns.');

        let analysisHTML = `
            <div class="symptom-analysis">
                <div class="severity-indicator severity-${urgencyLevel}">
                    <h4>Assessment Level: ${urgencyLevel.toUpperCase()}</h4>
                </div>
        `;

        if (warnings.length > 0) {
            analysisHTML += '<div class="warnings">';
            warnings.forEach(warning => {
                analysisHTML += `<p class="warning">${warning}</p>`;
            });
            analysisHTML += '</div>';
        }

        analysisHTML += `
                <div class="recommendations">
                    <h4>Recommendations:</h4>
                    <ul>
        `;

        recommendations.forEach(rec => {
            analysisHTML += `<li>${rec}</li>`;
        });

        analysisHTML += `
                    </ul>
                </div>
                <div class="disclaimer">
                    <p><strong>Disclaimer:</strong> This analysis is for informational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
                </div>
            </div>
        `;

        return analysisHTML;
    }

    scheduleAppointment() {
        const type = document.getElementById('appointmentType').value;
        const date = document.getElementById('preferredDate').value;
        const time = document.getElementById('preferredTime').value;
        const reason = document.getElementById('appointmentReason').value.trim();

        if (!type || !date || !time || !reason) {
            alert('Please fill in all fields to schedule an appointment.');
            return;
        }

        const appointment = {
            id: Date.now(),
            type: type,
            date: date,
            time: time,
            reason: reason,
            status: 'scheduled',
            created: new Date().toISOString()
        };

        this.appointments.push(appointment);
        this.saveAppointments();
        this.updateAppointmentsList();

        // Clear form
        document.getElementById('appointmentType').value = '';
        document.getElementById('preferredDate').value = '';
        document.getElementById('preferredTime').value = '';
        document.getElementById('appointmentReason').value = '';

        alert('Appointment scheduled successfully!');
    }

    updateAppointmentsList() {
        const listDiv = document.getElementById('appointmentsList');
        
        if (this.appointments.length === 0) {
            listDiv.innerHTML = '<p class="no-appointments">No upcoming appointments scheduled.</p>';
            return;
        }

        let html = '';
        this.appointments.forEach(appointment => {
            const formattedDate = new Date(appointment.date).toLocaleDateString();
            const timeFormatted = this.formatTime(appointment.time);
            
            html += `
                <div class="appointment-item">
                    <div class="appointment-details">
                        <h4>${appointment.type.charAt(0).toUpperCase() + appointment.type.slice(1)}</h4>
                        <p><i class="fas fa-calendar"></i> ${formattedDate} at ${timeFormatted}</p>
                        <p><i class="fas fa-notes-medical"></i> ${appointment.reason}</p>
                    </div>
                    <div class="appointment-actions">
                        <button class="btn-small btn-danger" onclick="healthBot.cancelAppointment(${appointment.id})">
                            <i class="fas fa-times"></i> Cancel
                        </button>
                    </div>
                </div>
            `;
        });

        listDiv.innerHTML = html;
    }

    cancelAppointment(appointmentId) {
        if (confirm('Are you sure you want to cancel this appointment?')) {
            this.appointments = this.appointments.filter(apt => apt.id !== appointmentId);
            this.saveAppointments();
            this.updateAppointmentsList();
        }
    }

    saveVitals() {
        const bloodPressure = document.getElementById('bloodPressure').value.trim();
        const heartRate = document.getElementById('heartRate').value;
        const temperature = document.getElementById('temperature').value;
        const weight = document.getElementById('weight').value;

        if (!bloodPressure && !heartRate && !temperature && !weight) {
            alert('Please enter at least one vital sign measurement.');
            return;
        }

        const vital = {
            id: Date.now(),
            date: new Date().toISOString(),
            bloodPressure: bloodPressure || null,
            heartRate: heartRate || null,
            temperature: temperature || null,
            weight: weight || null
        };

        this.vitals.push(vital);
        this.saveVitalsToStorage();

        // Clear form
        document.getElementById('bloodPressure').value = '';
        document.getElementById('heartRate').value = '';
        document.getElementById('temperature').value = '';
        document.getElementById('weight').value = '';

        alert('Vital signs saved successfully!');
    }

    addMedication() {
        const name = document.getElementById('medicationName').value.trim();
        const dosage = document.getElementById('dosage').value.trim();
        const frequency = document.getElementById('frequency').value;

        if (!name || !dosage || !frequency) {
            alert('Please fill in all medication fields.');
            return;
        }

        const medication = {
            id: Date.now(),
            name: name,
            dosage: dosage,
            frequency: frequency,
            added: new Date().toISOString()
        };

        this.medications.push(medication);
        this.saveMedications();
        this.updateMedicationsList();

        // Clear form
        document.getElementById('medicationName').value = '';
        document.getElementById('dosage').value = '';
        document.getElementById('frequency').value = 'once';
    }

    updateMedicationsList() {
        const listDiv = document.getElementById('medicationList');
        
        if (this.medications.length === 0) {
            listDiv.innerHTML = '<p class="no-medications">No medications added yet.</p>';
            return;
        }

        let html = '';
        this.medications.forEach(medication => {
            const frequencyText = {
                'once': 'Once daily',
                'twice': 'Twice daily',
                'thrice': 'Three times daily',
                'asneeded': 'As needed'
            };

            html += `
                <div class="medication-item">
                    <div class="medication-info">
                        <h4>${medication.name}</h4>
                        <p>${medication.dosage} - ${frequencyText[medication.frequency]}</p>
                    </div>
                    <button class="remove-medication" onclick="healthBot.removeMedication(${medication.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
        });

        listDiv.innerHTML = html;
    }

    removeMedication(medicationId) {
        if (confirm('Are you sure you want to remove this medication?')) {
            this.medications = this.medications.filter(med => med.id !== medicationId);
            this.saveMedications();
            this.updateMedicationsList();
        }
    }

    showHealthInfo(category) {
        const contentDiv = document.getElementById('infoContent');
        const healthInfo = this.getHealthInfoContent(category);
        contentDiv.innerHTML = healthInfo;
    }

    getHealthInfoContent(category) {
        const content = {
            general: `
                <h3><i class="fas fa-heart"></i> General Health Guidelines</h3>
                <div class="health-tips">
                    <h4>Daily Health Habits</h4>
                    <ul>
                        <li><strong>Stay Hydrated:</strong> Drink 8-10 glasses of water daily</li>
                        <li><strong>Balanced Diet:</strong> Include fruits, vegetables, whole grains, and lean proteins</li>
                        <li><strong>Regular Exercise:</strong> At least 150 minutes of moderate activity per week</li>
                        <li><strong>Adequate Sleep:</strong> 7-9 hours per night for adults</li>
                        <li><strong>Stress Management:</strong> Practice relaxation techniques</li>
                        <li><strong>Regular Check-ups:</strong> Annual physical exams and screenings</li>
                    </ul>
                    
                    <h4>Warning Signs to Watch For</h4>
                    <ul>
                        <li>Persistent fever or unusual symptoms</li>
                        <li>Unexplained weight loss or gain</li>
                        <li>Changes in energy levels or mood</li>
                        <li>New or worsening pain</li>
                    </ul>
                </div>
            `,
            nutrition: `
                <h3><i class="fas fa-apple-alt"></i> Nutrition Guidelines</h3>
                <div class="health-tips">
                    <h4>Balanced Diet Principles</h4>
                    <ul>
                        <li><strong>Vegetables & Fruits:</strong> Fill half your plate with colorful produce</li>
                        <li><strong>Whole Grains:</strong> Choose brown rice, quinoa, whole wheat bread</li>
                        <li><strong>Lean Proteins:</strong> Fish, poultry, beans, nuts, and seeds</li>
                        <li><strong>Healthy Fats:</strong> Olive oil, avocados, nuts</li>
                        <li><strong>Limit:</strong> Processed foods, added sugars, excessive sodium</li>
                    </ul>
                    
                    <h4>Portion Control Tips</h4>
                    <ul>
                        <li>Use smaller plates and bowls</li>
                        <li>Eat slowly and mindfully</li>
                        <li>Listen to your body's hunger cues</li>
                        <li>Stay hydrated - sometimes thirst feels like hunger</li>
                    </ul>
                </div>
            `,
            exercise: `
                <h3><i class="fas fa-dumbbell"></i> Exercise & Fitness</h3>
                <div class="health-tips">
                    <h4>Weekly Exercise Goals</h4>
                    <ul>
                        <li><strong>Cardio:</strong> 150 minutes moderate or 75 minutes vigorous activity</li>
                        <li><strong>Strength Training:</strong> 2+ days per week, all major muscle groups</li>
                        <li><strong>Flexibility:</strong> Daily stretching or yoga</li>
                        <li><strong>Balance:</strong> Especially important for older adults</li>
                    </ul>
                    
                    <h4>Getting Started Safely</h4>
                    <ul>
                        <li>Start slowly and gradually increase intensity</li>
                        <li>Warm up before and cool down after exercise</li>
                        <li>Listen to your body and rest when needed</li>
                        <li>Consult a doctor before starting intense programs</li>
                        <li>Stay hydrated during workouts</li>
                    </ul>
                </div>
            `,
            mental: `
                <h3><i class="fas fa-brain"></i> Mental Health & Wellness</h3>
                <div class="health-tips">
                    <h4>Mental Health Strategies</h4>
                    <ul>
                        <li><strong>Stress Management:</strong> Deep breathing, meditation, yoga</li>
                        <li><strong>Social Connection:</strong> Maintain relationships with family and friends</li>
                        <li><strong>Purpose & Meaning:</strong> Engage in activities you find fulfilling</li>
                        <li><strong>Professional Help:</strong> Don't hesitate to seek counseling or therapy</li>
                        <li><strong>Healthy Boundaries:</strong> Learn to say no and manage commitments</li>
                    </ul>
                    
                    <h4>Warning Signs</h4>
                    <ul>
                        <li>Persistent sadness or anxiety</li>
                        <li>Changes in sleep or appetite</li>
                        <li>Loss of interest in activities</li>
                        <li>Difficulty concentrating</li>
                        <li>Thoughts of self-harm</li>
                    </ul>
                    
                    <p><strong>Crisis Resources:</strong><br>
                    National Suicide Prevention Lifeline: 988<br>
                    Crisis Text Line: Text HOME to 741741</p>
                </div>
            `,
            prevention: `
                <h3><i class="fas fa-shield-alt"></i> Preventive Care</h3>
                <div class="health-tips">
                    <h4>Regular Screenings</h4>
                    <ul>
                        <li><strong>Blood Pressure:</strong> At least every 2 years</li>
                        <li><strong>Cholesterol:</strong> Every 4-6 years starting at age 20</li>
                        <li><strong>Diabetes:</strong> Every 3 years starting at age 45</li>
                        <li><strong>Cancer Screenings:</strong> As recommended by age and risk factors</li>
                        <li><strong>Eye Exams:</strong> Every 1-2 years</li>
                        <li><strong>Dental:</strong> Every 6 months</li>
                    </ul>
                    
                    <h4>Vaccinations</h4>
                    <ul>
                        <li>Stay up to date with routine vaccines</li>
                        <li>Annual flu vaccination</li>
                        <li>COVID-19 vaccines as recommended</li>
                        <li>Travel vaccines when needed</li>
                    </ul>
                </div>
            `,
            emergency: `
                <h3><i class="fas fa-first-aid"></i> First Aid & Emergency Care</h3>
                <div class="health-tips">
                    <h4>When to Call 911</h4>
                    <ul>
                        <li>Chest pain or pressure</li>
                        <li>Difficulty breathing or shortness of breath</li>
                        <li>Severe bleeding that won't stop</li>
                        <li>Loss of consciousness</li>
                        <li>Severe allergic reaction</li>
                        <li>Signs of stroke (F.A.S.T.)</li>
                        <li>Severe burns</li>
                        <li>Suspected poisoning</li>
                    </ul>
                    
                    <h4>Basic First Aid Tips</h4>
                    <ul>
                        <li><strong>Cuts:</strong> Apply direct pressure to stop bleeding</li>
                        <li><strong>Burns:</strong> Cool with running water for 10-20 minutes</li>
                        <li><strong>Choking:</strong> Perform Heimlich maneuver</li>
                        <li><strong>Sprains:</strong> Rest, Ice, Compression, Elevation (R.I.C.E.)</li>
                    </ul>
                    
                    <h4>Emergency Kit Essentials</h4>
                    <ul>
                        <li>Bandages and gauze</li>
                        <li>Antiseptic wipes</li>
                        <li>Pain relievers</li>
                        <li>Thermometer</li>
                        <li>Emergency contact numbers</li>
                    </ul>
                </div>
            `
        };

        return content[category] || '<p>Information not available.</p>';
    }

    showEmergencyModal() {
        document.getElementById('emergencyModal').style.display = 'block';
    }

    hideEmergencyModal() {
        document.getElementById('emergencyModal').style.display = 'none';
    }

    // Utility functions
    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    formatTime(time) {
        const [hours, minutes] = time.split(':');
        const hour = parseInt(hours);
        const ampm = hour >= 12 ? 'PM' : 'AM';
        const displayHour = hour % 12 || 12;
        return `${displayHour}:${minutes} ${ampm}`;
    }

    updateTimestamps() {
        document.querySelectorAll('.timestamp').forEach(timestamp => {
            if (timestamp.textContent === '') {
                timestamp.textContent = this.getCurrentTime();
            }
        });
    }

    // Storage functions
    saveAppointments() {
        localStorage.setItem('appointments', JSON.stringify(this.appointments));
    }

    saveMedications() {
        localStorage.setItem('medications', JSON.stringify(this.medications));
    }

    saveVitalsToStorage() {
        localStorage.setItem('vitals', JSON.stringify(this.vitals));
    }

    loadStoredData() {
        this.updateAppointmentsList();
        this.updateMedicationsList();
    }
}

// Global functions for onclick handlers
function findNearestHospital() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const url = `https://www.google.com/maps/search/hospital+near+me/@${lat},${lon},15z`;
            window.open(url, '_blank');
        }, function() {
            window.open('https://www.google.com/maps/search/hospital+near+me', '_blank');
        });
    } else {
        window.open('https://www.google.com/maps/search/hospital+near+me', '_blank');
    }
}

// Initialize the application
let healthBot;
document.addEventListener('DOMContentLoaded', function() {
    healthBot = new HealthBot();
});

// Service Worker for offline functionality (optional enhancement)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HealthBot;
}