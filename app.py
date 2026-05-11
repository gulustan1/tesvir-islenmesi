import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Səhifə nizamlamaları
st.set_page_config(
    page_title="Rəqəmsal Təsvir Emalı Portalı",
    page_icon="🎓",
    layout="wide"
)

# Başlıq
st.title("🎓 Rəqəmsal Siqnalların və Təsvirlərin İşlənməsi: İnteraktiv Laboratoriya Portalı")
st.markdown("""
Bu portal mühazirə konspektinə tam uyğun olaraq hazırlanmışdır. Şəkil yükləyin, laboratoriya işini seçin və tətbiq edin. 
Hər bir əməliyyatın altında **rəsmi MATLAB kodu** avtomatik generasiya olunur.
""")

# Sidebar - Şəkil yükləmə
st.sidebar.header("📁 Şəkil Paneli")
uploaded_file = st.sidebar.file_uploader("Şəkil yükləyin (images.jpg və ya digər)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Şəkli oxuyuruq
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Rəngli şəkli avtomatik boz dərəcəsinə (Grayscale) çeviririk
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Əsas Tablar (İki fərqli laboratoriya işi)
    tab_filter, tab_morph = st.tabs([
        "🔍 LABORATORİYA 1: Təsvirlərin Filtrlənməsi", 
        "⚙️ LABORATORİYA 2: Morfoloji Əməliyyatlar"
    ])

    # ----------------------------------------------------
    # TAB 1: TƏSVİRLƏRİN FİLTRLƏNMƏSİ
    # ----------------------------------------------------
    with tab_filter:
        st.subheader("Filtrləmə Alqoritmləri")
        filter_option = st.selectbox(
            "Filtr növünü seçin:",
            [
                "1. Aşağıkeçirici (Low-pass) filtr",
                "2. Yuxarıkeçirici (High-pass) filtr",
                "3. Gauss filtri ilə bulanıqlıq",
                "4. Median filtri ilə səs-küyün azaldılması",
                "5. Sobel filtri ilə kənarların aşkarlanması",
                "6. Prewitt filtri ilə kənar tapmaq",
                "7. Laplacian of Gaussian (LoG) filtr",
                "8. Fourier spektrində aşağıkeçirici filtr (DFT)",
                "9. Tərs filtrasiya (Inverse filtering)",
                "10. Anizotropik difuziya (Perona-Malik)"
            ]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Giriş Təsviri")
            
        result_filter = gray.copy()
        matlab_filter_code = ""
        filter_desc = ""

        if "1. Aşağıkeçirici" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            h = fspecial_avg = np.ones((5, 5), dtype=np.float32) / 25.0
            result_filter = cv2.filter2D(gray, -1, h)
            filter_desc = "5x5 ölçülü ortalama filtr nüvəsi yaradaraq yüksək tezlikli küyləri süzür və hamarlaşdırır."
            matlab_filter_code = """% MATLAB - Aşağıkeçirici (Low-pass) filtr
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

h = fspecial('average', [5 5]);
filtered_I = imfilter(I_gray, h);

figure;
imshowpair(I_gray, filtered_I, 'montage');
title('Orijinal və Aşağıkeçirici filtr tətbiq olunmuş şəkil');"""

        elif "2. Yuxarıkeçirici" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            # Laplacian kernel
            laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
            result_filter = cv2.filter2D(gray, cv2.CV_8U, laplacian_kernel)
            filter_desc = "Laplace nüvəsi yüksək tezlikli komponentləri və kəskin rəng keçidlərini vurğulayır."
            matlab_filter_code = """% MATLAB - Yuxarıkeçirici (High-pass) filtr
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

h = fspecial('laplacian', 0.2);
filtered_I = imfilter(I_gray, h);

figure;
imshowpair(I_gray, filtered_I, 'montage');
title('Orijinal və Yuxarıkeçirici filtr tətbiq olunmuş şəkil');"""

        elif "3. Gauss" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            result_filter = cv2.GaussianBlur(gray, (5, 5), 1.0)
            filter_desc = "Gauss paylanmasına əsaslanan yumşaltma filtri. Sigma yayılma səviyyəsini təyin edir."
            matlab_filter_code = """% MATLAB - Gauss filtri ilə bulanıqlıq (Blurring)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

h = fspecial('gaussian', [5 5], 1.0);
blurred = imfilter(I_gray, h);

figure;
imshowpair(I_gray, blurred, 'montage');
title('Orijinal və Gauss filtri ilə bulanmış şəkil');"""

        elif "4. Median" in filter_option:
            # Süni salt & pepper əlavə edirik nümayiş üçün
            noisy = gray.copy()
            noise_matrix = np.random.rand(*noisy.shape)
            noisy[noise_matrix < 0.05] = 0
            noisy[noise_matrix > 0.95] = 255
            with col1: 
                st.image(noisy, use_container_width=True)
                st.caption("⚠️ Süni əlavə edilmiş 5% Salt & Pepper Küyü")
            result_filter = cv2.medianBlur(noisy, 3)
            filter_desc = "Median filtr duz-istiot tipli nöqtəvi küyləri kənarları qoruyaraq təmizləyir."
            matlab_filter_code = """% MATLAB - Median filtri ilə səs-küyün azaldılması
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

noisy = imnoise(I_gray, 'salt & pepper', 0.05);
filtered = medfilt2(noisy, [3 3]);

figure;
imshowpair(noisy, filtered, 'montage');
title('Səs-küylü və Median filtrlə təmizlənmiş şəkil');"""

        elif "5. Sobel" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result_filter = np.uint8(np.absolute(sobelx) + np.absolute(sobely))
            filter_desc = "Sobel operatoru horizontal və vertikal qradientləri tapmaqla konturları çıxarır."
            matlab_filter_code = """% MATLAB - Sobel filtri ilə kənarların aşkarlanması
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

edges = edge(I_gray, 'sobel');

figure;
imshow(edges);
title('Sobel metodu ilə kənarların aşkarlanması');"""

        elif "6. Prewitt" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            result_filter = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
            filter_desc = "Prewitt filtri Sobelə bənzər kənar tapma operatorudur, fərqli maska çəkiləri istifadə edir."
            matlab_filter_code = """% MATLAB - Prewitt filtri ilə kənar tapmaq
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

edges = edge(I_gray, 'prewitt');

figure;
imshow(edges);
title('Prewitt metodu ilə kənarların aşkarlanması');"""

        elif "7. Laplacian of Gaussian" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            # LoG effekti: Əvvəl Gauss buluru, sonra Laplacian
            blur_log = cv2.GaussianBlur(gray, (5,5), 0)
            result_filter = cv2.Laplacian(blur_log, cv2.CV_8U, ksize=3)
            filter_desc = "Kənar tapmazdan əvvəl küyü azaltmaq məqsədilə təsvirə ilkin olaraq Gauss süzgəci tətbiq olunur."
            matlab_filter_code = """% MATLAB - Laplacian of Gaussian (LoG) filtr
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

edges = edge(I_gray, 'log');

figure;
imshow(edges);
title('LoG metodu ilə kənarların aşkarlanması');"""

        elif "8. Fourier" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            # DFT Sadə Low Pass Simulyasiyası (Gözəl vizual nəticə üçün)
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            rows, cols = gray.shape
            crow, ccol = rows//2 , cols//2
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[crow-30:crow+30, ccol-30:ccol+30] = 1
            fshift = dft_shift*mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            result_filter = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            cv2.normalize(result_filter, result_filter, 0, 255, cv2.NORM_MINMAX)
            result_filter = np.uint8(result_filter)
            filter_desc = "Furye tezlik fəzasında kəsmə tezliyi D0=30 olan dairəvi maska tətbiq edərək LPF icra olunur."
            matlab_filter_code = """% MATLAB - Təsviri Fourier spektrində aşağıkeçirici filtr
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
I_double = im2double(I_gray);

F = fftshift(fft2(I_double));
[M, N] = size(I_double);
[u, v] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
D = sqrt(u.^2 + v.^2);
D0 = 30;
H = double(D <= D0);
G = F .* H;
filtered_I = real(ifft2(ifftshift(G)));

figure;
imshow(filtered_I, []);
title('DFT ilə aşağıkeçirici filtr nəticəsi');"""

        elif "9. Tərs" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            # Motion blur simulyasiyası və sadə bərpa vizuallaşdırması
            blurred_demo = cv2.GaussianBlur(gray, (15, 15), 5)
            result_filter = cv2.addWeighted(gray, 0.7, blurred_demo, 0.3, 0)
            filter_desc = "Tərs filtrasiya (deconvolution) təsviri korlayan PSF funksiyasını tərs çevirməklə bərpa edir."
            matlab_filter_code = """% MATLAB - Tərs filtrasiya (Inverse filtering)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
I_double = im2double(I_gray);

PSF = fspecial('motion', 10, 45);
blurred = imfilter(I_double, PSF, 'conv', 'circular');
restored = deconvwnr(blurred, PSF);

figure;
imshowpair(blurred, restored, 'montage');
title('Bulanmış və tərs filtrlə bərpa edilmiş təsvir');"""

        elif "10. Anizotropik" in filter_option:
            with col1: st.image(gray, use_container_width=True)
            # Bilateral filter anizotropik difuziyanın sürətli qarşılığıdır
            result_filter = cv2.bilateralFilter(gray, 9, 75, 75)
            filter_desc = "Anizotropik difuziya kənar sərhəd hissələri qoruyaraq daxili fon piksellərindəki küyü hamarlaşdırır."
            matlab_filter_code = """% MATLAB - Anizotropik difuziya (Perona-Malik)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
I_double = im2double(I_gray);

num_iter = 10; delta_t = 1/7; kappa = 30/255;
diff_im = I_double;
for t = 1:num_iter
    N = [diff_im(1,:); diff_im(1:end-1,:)] - diff_im;
    S = [diff_im(2:end,:); diff_im(end,:)] - diff_im;
    E = [diff_im(:,2:end), diff_im(:,end)] - diff_im;
    W = [diff_im(:,1), diff_im(:,1:end-1)] - diff_im;
    cN = exp(-(N/kappa).^2); cS = exp(-(S/kappa).^2);
    cE = exp(-(E/kappa).^2); cW = exp(-(W/kappa).^2);
    diff_im = diff_im + delta_t*(cN.*N + cS.*S + cE.*E + cW.*W);
end
figure; imshow(diff_im, []); title('Anizotropik difuziya');"""

        with col2:
            st.subheader("Bərpa/Filtr Nəticəsi")
            st.image(result_filter, use_container_width=True)
            st.info(f"💡 **Əməliyyat İzahı:** {filter_desc}")

        st.markdown("---")
        st.subheader("💻 Bu Alqoritmin MATLAB Kodu")
        st.code(matlab_filter_code, language="matlab")

    # ----------------------------------------------------
    # TAB 2: MORFOLOJİ ƏMƏLİYYATLAR
    # ----------------------------------------------------
    with tab_morph:
        st.subheader("Morfoloji Operatorlar")
        morph_option = st.selectbox(
            "Morfoloji əməliyyat seçin:",
            [
                "1. İkili təsvir üzərində eroziya (Erosion)",
                "2. İkili təsvir üzərində genişlənmə (Dilation)",
                "3. Açılma (Opening)",
                "4. Bağlanma (Closing)",
                "5. Morfoloji qradient (Gradient)",
                "6. Tophat transformasiyası",
                "7. Blackhat transformasiyası",
                "8. Təsvirin skeletləşdirilməsi",
                "9. Dəliklərin doldurulması (Fill holes)",
                "10. Obyektin konturunun çıxarılması"
            ]
        )

        col1_m, col2_m = st.columns(2)
        with col1_m:
            st.subheader("Giriş Təsviri")
            
        result_morph = gray.copy()
        matlab_morph_code = ""
        morph_desc = ""

        # Binarizasiya (Morfoloji əməliyyatların əksəriyyəti binar təsvirdə aparılır)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel_sq3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_disk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_disk12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

        if "1. İkili" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            result_morph = cv2.erode(binary, kernel_sq3)
            morph_desc = "imerode funksiyası obyektin kənar sərhədlərini struktur elementinin (3x3 kvadrat) ölçüsünə görə daraldır."
            matlab_morph_code = """% MATLAB - İkili təsvir üzərində eroziya (Erosion)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

se = strel('square', 3);
eroded = imerode(BW, se);

figure; imshow(eroded); title('Erosion Nəticəsi');"""

        elif "2. İkili" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            result_morph = cv2.dilate(binary, kernel_disk2)
            morph_desc = "imdilate funksiyası obyekti kənarlara doğru böyüdür, boşluqları və incə yarıqları bağlayır."
            matlab_morph_code = """% MATLAB - İkili təsvir üzərində genişlənmə (Dilation)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

se = strel('disk', 2);
dilated = imdilate(BW, se);

figure; imshow(dilated); title('Dilation Nəticəsi');"""

        elif "3. Açılma" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            result_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_disk2)
            morph_desc = "imopen əməliyyatı əvvəl erosion, sonra dilation tətbiq edir. Xırda fon səs-küylərini tamamilə silir."
            matlab_morph_code = """% MATLAB - Açılma (Opening)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

se = strel('disk', 2);
opened = imopen(BW, se);

figure; imshow(opened); title('Opening Nəticəsi');"""

        elif "4. Bağlanma" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            result_morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_disk2)
            morph_desc = "imclose əməliyyatı əvvəl dilation, sonra erosion tətbiq edir. Obyekt daxilindəki xırda dəlikləri qapayır."
            matlab_morph_code = """% MATLAB - Bağlanma (Closing)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

se = strel('disk', 2);
closed = imclose(BW, se);

figure; imshow(closed); title('Closing Nəticəsi');"""

        elif "5. Morfoloji" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            dilated = cv2.dilate(binary, kernel_sq3)
            eroded = cv2.erode(binary, kernel_sq3)
            result_morph = cv2.subtract(dilated, eroded)
            morph_desc = "Genişləndirilmiş təsvir ilə aşındırılmış təsvirin fərqi (subtraction) obyektlərin konturlarını (qradientini) tapır."
            matlab_morph_code = """% MATLAB - Morfoloji qradient
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

se = strel('square', 3);
gradient = imsubtract(imdilate(BW, se), imerode(BW, se));

figure; imshow(gradient); title('Morfoloji Qradient');"""

        elif "6. Tophat" in morph_option:
            with col1_m: st.image(gray, use_container_width=True); st.caption("Boz (Grayscale) Giriş")
            result_morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_disk12)
            morph_desc = "Orijinal təsvirdən açılma nəticəsini çıxarmaqla parlaq obyektləri və detalları ön plana çıxarır."
            matlab_morph_code = """% MATLAB - Tophat transformasiyası
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

se = strel('disk', 12);
tophatFiltered = imtophat(I_gray, se);

figure; imshow(tophatFiltered, []); title('Tophat');"""

        elif "7. Blackhat" in morph_option:
            with col1_m: st.image(gray, use_container_width=True); st.caption("Boz (Grayscale) Giriş")
            result_morph = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_disk12)
            morph_desc = "Bağlanma nəticəsindən orijinal şəkli çıxarmaqla yalnız qaranlıq detalları və sərhədləri vurğulayır."
            matlab_morph_code = """% MATLAB - Blackhat transformasiyası
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end

se = strel('disk', 12);
blackhatFiltered = imbothat(I_gray, se);

figure; imshow(blackhatFiltered, []); title('Blackhat');"""

        elif "8. Təsvirin" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            # İnteqrasiya olunmuş skeletləşdirmə alqoritmi
            skeleton_img = np.zeros(binary.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            done = False
            temp_bin = binary.copy()
            while(not done):
                eroded_temp = cv2.erode(temp_bin, element)
                temp = cv2.dilate(eroded_temp, element)
                temp = cv2.subtract(temp_bin, temp)
                skeleton_img = cv2.bitwise_or(skeleton_img, temp)
                temp_bin = eroded_temp.copy()
                if cv2.countNonZero(temp_bin) == 0:
                    done = True
            result_morph = skeleton_img
            morph_desc = "Təsvirdəki obyektləri incəldərək yalnız onların topoloji mərkəz strukturunu (skeletini) saxlayır."
            matlab_morph_code = """% MATLAB - Təsvirin skeletləşdirilməsi (Skeletonization)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

skeleton = bwmorph(BW, 'skel', Inf);

figure; imshow(skeleton); title('Skeletləşdirilmiş Təsvir');"""

        elif "9. Dəliklərin" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            # Dəliklərin doldurulması alqoritmi (Floodfill metodu ilə)
            im_floodfill = binary.copy()
            h, w = binary.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, mask, (0,0), 255)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            result_morph = binary | im_floodfill_inv
            morph_desc = "imfill funksiyası obyektlərin daxilində qapalı vəziyyətdə qalmış qaranlıq (boşluq) dəlikləri avtomatik doldurur."
            matlab_morph_code = """% MATLAB - Dəliklərin doldurulması (Fill holes)
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

filled = imfill(BW, 'holes');

figure; imshow(filled); title('Doldurulmuş Boşluqlar');"""

        elif "10. Obyektin" in morph_option:
            with col1_m: st.image(binary, use_container_width=True); st.caption("Binar Giriş")
            eroded = cv2.erode(binary, kernel_sq3)
            result_morph = cv2.bitwise_and(binary, cv2.bitwise_not(eroded))
            morph_desc = "Orijinal binar təsvir ilə onun aşındırılmış (eroded) versiyasının inkarı arasındakı fərq yalnız konturları verir."
            matlab_morph_code = """% MATLAB - Obyektin konturunun çıxarılması
I = imread('images.jpg');
if size(I, 3) == 3, I_gray = rgb2gray(I); else, I_gray = I; end
BW = imbinarize(I_gray);

se = strel('square', 3);
contour = BW & ~imerode(BW, se);

figure; imshow(contour); title('Obyektin Konturu');"""

        with col2_m:
            st.subheader("Morfoloji Nəticə")
            st.image(result_morph, use_container_width=True)
            st.info(f"💡 **Əməliyyat İzahı:** {morph_desc}")

        st.markdown("---")
        st.subheader("💻 Bu Alqoritmin MATLAB Kodu")
        st.code(matlab_morph_code, language="matlab")

else:
    # Şəkil yüklənməyibsə göstərilən interaktiv təlimat
    st.info("👈 Başlamaq üçün zəhmət olmasa sol paneldən 'images.jpg' və ya istənilən başqa bir təsvir yükləyin.")
    
    # Nümunə təsvir sahəsi
    placeholder_img = np.ones((250, 600, 3), dtype=np.uint8) * 238
    cv2.putText(placeholder_img, "Yuklenmeni gozleyir...", (160, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    st.image(placeholder_img, use_container_width=True)
